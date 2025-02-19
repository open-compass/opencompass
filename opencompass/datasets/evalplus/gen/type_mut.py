import copy
import random
import string
import time
from typing import Any, Dict, List, Set, Tuple

from multipledispatch import dispatch

from evalplus.gen.mut_gen import MutateGen
from evalplus.gen.util import trusted_check_exec

MAX_MULTI_STEP_SIZE = 5
MUTATE_BOUND_SIZE = 8

NoneType = type(None)


# decorator to use ingredients
class use_ingredient:
    def __init__(self, prob: float):
        assert 0 <= prob <= 0.95
        self.prob = prob

    def __call__(obj, func):
        def wrapper(self, seed_input):
            if random.random() < obj.prob and self.ingredients[type(seed_input)]:
                return random.choice(list(self.ingredients[type(seed_input)]))
            else:
                return func(self, seed_input)

        return wrapper


class TypedMutGen(MutateGen):
    def __init__(self, inputs: List, signature: str, contract_code: str):
        super().__init__(inputs, signature, contract_code)
        self.timeout = 60 * 60  # 1 hour
        self.ingredients = {
            int: set(),
            float: set(),
            str: set(),
            complex: set(),
        }
        for x in inputs:
            self.fetch_ingredient(x)

    def seed_selection(self):
        # random for now.
        return random.choice(self.seed_pool)

    def mutate(self, seed_input: Any) -> List:
        new_input = copy.deepcopy(seed_input)

        patience = MUTATE_BOUND_SIZE
        while new_input == seed_input or patience == 0:
            new_input = self.typed_mutate(new_input)
            patience -= 1

        return new_input

    #########################
    # Type-aware generation #
    #########################
    @dispatch(NoneType)
    def typed_gen(self, _):
        return None

    @dispatch(int)
    def typed_gen(self, _):
        @use_ingredient(0.5)
        def _impl(*_):
            return random.randint(-100, 100)

        return _impl(self, _)

    @dispatch(float)
    def typed_gen(self, _):
        @use_ingredient(0.5)
        def _impl(*_):
            return random.uniform(-100, 100)

        return _impl(self, _)

    @dispatch(bool)
    def typed_gen(self, _):
        return random.choice([True, False])

    @dispatch(str)
    def typed_gen(self, _):
        @use_ingredient(0.5)
        def _impl(*_):
            return "".join(
                random.choice(string.ascii_letters)
                for _ in range(random.randint(0, 10))
            )

        return _impl(self, _)

    def any_gen(self):
        # weighted choose
        choice = random.choices(
            [
                True,
                1,
                1.1,
                "str",
                [],  # list
                tuple(),  # tuple
                dict(),  # dict
                None,  # None
            ],
            [0.2, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05],
        )[0]
        return self.typed_gen(choice)

    @dispatch(list)
    def typed_gen(self, _):
        ret = []
        size = random.randint(0, 10)
        if random.randint(0, 4) == 0:  # heterogeneous
            for _ in range(size):
                ret.append(self.any_gen())
        else:  # homogeneous
            t = random.choice([bool(), int(), float(), str()])
            for _ in range(size):
                ret.append(self.typed_gen(t))
        return ret

    @dispatch(tuple)
    def typed_gen(self, _):
        return tuple(self.typed_gen([]))

    # NOTE: disable set for now as Steven is too weak in Python (/s)
    # @dispatch(set)
    # def typed_gen(self, _):
    #     return set(self.typed_gen([]))

    @dispatch(dict)
    def typed_gen(self, _):
        ret = dict()
        values = self.typed_gen([])
        # NOTE: Assumption: nobody uses dict with heterogeneous keys
        # NOTE: Assumption: nobody uses dict with boolean keys
        key_type = random.choice([int(), float(), str()])
        for v in values:
            ret[self.typed_gen(key_type)] = self.typed_gen(v)
        return ret

    ########################
    # Type-aware mutation  #
    ########################
    # Simple primitives
    @dispatch(int)
    def typed_mutate(self, seed_input: int):
        @use_ingredient(0.5)
        def _impl(_, seed_input: int):
            return seed_input + random.randint(-1, 1)

        return _impl(self, seed_input)

    @dispatch(float)
    def typed_mutate(self, seed_input: float):
        @use_ingredient(0.5)
        def _impl(_, seed_input: float):
            if random.randint(0, 1):
                return seed_input + random.uniform(-1, 1)
            return seed_input * (1 + random.uniform(-0.5, 0.5))

        return _impl(self, seed_input)

    @dispatch(complex)
    def typed_mutate(self, seed_input: complex):
        @use_ingredient(0.5)
        def _impl(_, seed_input: complex):
            imag = seed_input.imag + random.uniform(-1, 1)
            return complex(0, imag)

        return _impl(self, seed_input)

    @dispatch(bool)
    def typed_mutate(self, seed_input: bool):
        return random.choice([True, False])

    @dispatch(NoneType)
    def typed_mutate(self, seed_input: NoneType):
        return None

    # List-like
    @dispatch(list)
    def typed_mutate(self, seed_input: List):
        if len(seed_input) == 0:
            return self.typed_gen([])

        choice = random.randint(0, 3)
        idx = random.randint(0, len(seed_input) - 1)
        if choice == 0:  # remove one element
            seed_input.pop(random.randint(0, len(seed_input) - 1))
        elif choice == 1 and len(seed_input) > 0:  # add one mutated element
            seed_input.insert(
                random.randint(0, len(seed_input) - 1),
                self.typed_mutate(seed_input[idx]),
            )
        elif choice == 2 and len(seed_input) > 0:  # repeat one element
            seed_input.append(seed_input[idx])
        else:  # inplace element change
            seed_input[idx] = self.typed_mutate(seed_input[idx])
        return seed_input

    @dispatch(tuple)
    def typed_mutate(self, seed_input: Tuple):
        return tuple(self.typed_mutate(list(seed_input)))

    # String
    @dispatch(str)
    def typed_mutate(self, seed_input: str):
        @use_ingredient(0.4)
        def _impl(_, seed_input: str):
            choice = random.randint(0, 2) if seed_input else 0
            if choice == 0 and self.ingredients[str]:  # insert an ingredient
                idx = random.randint(0, len(seed_input))
                return (
                    seed_input[:idx]
                    + random.choice(list(self.ingredients[str]))
                    + seed_input[idx:]
                )
            # other choices assume len(seed_input) > 0
            elif choice == 1:  # replace a substring with empty or mutated string
                start = random.randint(0, len(seed_input) - 1)
                end = random.randint(start + 1, len(seed_input))
                mid = (
                    ""
                    if random.randint(0, 1)
                    else self.typed_mutate(seed_input[start:end])
                )
                return seed_input[:start] + mid + seed_input[end:]
            elif choice == 2:  # repeat one element
                idx = random.randint(0, len(seed_input) - 1)
                return (
                    seed_input[:idx]
                    + seed_input[random.randint(0, len(seed_input) - 1)]
                    + seed_input[idx:]
                )

            # random char
            return self.typed_gen(str())

        return _impl(self, seed_input)

    # Set
    @dispatch(set)
    def typed_mutate(self, seed_input: Set):
        return set(self.typed_mutate(list(seed_input)))

    # Dict
    @dispatch(dict)
    def typed_mutate(self, seed_input: Dict):
        if len(seed_input) == 0:
            return self.typed_gen(dict())

        choice = random.randint(0, 2)
        if choice == 0:  # remove a kv
            del seed_input[random.choice(list(seed_input.keys()))]
        elif choice == 1:  # add a kv
            k = self.typed_mutate(random.choice(list(seed_input.keys())))
            v = self.typed_mutate(random.choice(list(seed_input.values())))
            seed_input[k] = v
        elif choice == 2:  # inplace value change
            k0, v0 = random.choice(list(seed_input.items()))
            seed_input[k0] = self.typed_mutate(v0)
        return seed_input

    ############################################
    # Fetching ingredients to self.ingredients #
    ############################################
    def fetch_ingredient(self, seed_input):
        self.typed_fetch(seed_input)

    @dispatch(int)
    def typed_fetch(self, seed_input: int):
        self.ingredients[int].add(seed_input)

    @dispatch(float)
    def typed_fetch(self, seed_input: float):
        self.ingredients[float].add(seed_input)

    @dispatch(complex)
    def typed_fetch(self, seed_input: complex):
        self.ingredients[complex].add(seed_input)

    @dispatch(str)
    def typed_fetch(self, seed_input: str):
        self.ingredients[str].add(seed_input)
        for token in seed_input.strip().split():
            self.ingredients[str].add(token)

    # List-like
    def _fetch_list_like(self, seed_input):
        for x in seed_input:
            if self.typed_fetch.dispatch(type(x)):
                self.fetch_ingredient(x)

    @dispatch(list)
    def typed_fetch(self, seed_input: List):
        self._fetch_list_like(seed_input)

    @dispatch(tuple)
    def typed_fetch(self, seed_input: Tuple):
        self._fetch_list_like(seed_input)

    # NOTE: disable set for now as Steven is too weak in Python (/s)
    # @dispatch(set)
    # def typed_fetch(self, seed_input: Set):
    #     self._fetch_list_like(seed_input)

    # Dict
    @dispatch(dict)
    def typed_fetch(self, seed_input: Dict):
        self._fetch_list_like(seed_input.keys())
        self._fetch_list_like(seed_input.values())

    def generate(self, num: int):
        start = time.time()
        num_generated = 1
        while len(self.new_inputs) < num and time.time() - start < self.timeout:
            if num_generated % 1000 == 0:
                print(
                    f"generated {num_generated} already with {len(self.new_inputs)} new inputs ... "
                )
            new_input = self.seed_selection()
            # Multi-step instead of single-step
            for _ in range(random.randint(1, MAX_MULTI_STEP_SIZE)):
                new_input = self.mutate(new_input)
            num_generated += 1
            if hash(str(new_input)) not in self.seed_hash:
                if trusted_check_exec(self.contract, [new_input], self.entry_point):
                    self.typed_fetch(new_input)
                    self.seed_pool.append(new_input)
                    self.new_inputs.append(new_input)
                self.seed_hash.add(hash(str(new_input)))
        return self.new_inputs[:num]
