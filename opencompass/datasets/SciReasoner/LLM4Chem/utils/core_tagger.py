def find_sub_sequence(whole, sub):
    assert isinstance(whole, list)
    assert isinstance(sub, list)
    len_whole = len(whole)
    len_sub = len(sub)
    assert len_whole > 0
    assert len_sub > 0

    s = 0
    while True:
        s_whole = whole[s:]
        try:
            k_pos = s_whole.index(sub[0])
        except ValueError:
            return -1

        fail = False
        for i in range(1, len_sub):
            try:
                if s_whole[k_pos + i] != sub[i]:
                    fail = True
                    break
            except IndexError:
                return -1
        if fail:
            s = s + k_pos + 1
            continue
        else:
            return s + k_pos


class CoreTagger(object):

    def __init__(self,
                 tokenizer,
                 core_tags_as_special_tokens=False,
                 include_tags=True):
        self.tokenizer = tokenizer
        if core_tags_as_special_tokens:
            raise NotImplementedError
        self.core_tags_as_special_tokens = core_tags_as_special_tokens
        if not include_tags:
            raise NotImplementedError
        self.include_tags = include_tags

        self.left_tag_to_id = {}
        self.right_tag_to_id = {}

    def generate_mask(self, token_ids, output_begin, sample):
        mask = [0] * len(token_ids)
        left_tag, right_tag = sample['output_core_tag_left'], sample[
            'output_core_tag_right']
        if left_tag not in self.left_tag_to_id:
            if left_tag is None:
                left_token_ids = None
            else:
                left_token_ids = self.tokenizer(
                    left_tag,
                    add_special_tokens=False,
                    return_attention_mask=False)['input_ids']
            self.left_tag_to_id[left_tag] = left_token_ids
        else:
            left_token_ids = self.left_tag_to_id[left_tag]
        if right_tag not in self.right_tag_to_id:
            if right_tag is None:
                right_token_ids = None
            else:
                right_token_ids = self.tokenizer(
                    right_tag,
                    add_special_tokens=False,
                    return_attention_mask=False)['input_ids']
            self.right_tag_to_id[right_tag] = right_token_ids
        else:
            right_token_ids = self.right_tag_to_id[right_tag]

        output_token_ids = token_ids[output_begin:]
        if left_token_ids is None:
            left_position = output_begin
        elif len(output_token_ids) == 0:
            left_position = None
        else:
            left_position = find_sub_sequence(output_token_ids,
                                              left_token_ids) + output_begin
            if left_position == -1:
                left_position = None

        if left_position is None:
            return mask

        if right_token_ids is None:
            right_position = len(token_ids)
            if token_ids[-1] == self.tokenizer.eos_token_id:
                right_position -= 1
        else:
            right_position = find_sub_sequence(output_token_ids,
                                               right_token_ids) + output_begin
            if right_position == -1:
                right_position = len(token_ids)
                if token_ids[-1] == self.tokenizer.eos_token_id:
                    right_position -= 1
            else:
                right_position = min(right_position + len(right_token_ids),
                                     len(token_ids))

        for idx in range(left_position, right_position):
            mask[idx] = 1

        return mask


class CoreTaggerGeneral(object):

    def __init__(self,
                 tokenizer,
                 core_tags_as_special_tokens=False,
                 include_tags=True):
        self.tokenizer = tokenizer
        if core_tags_as_special_tokens:
            raise NotImplementedError
        self.core_tags_as_special_tokens = core_tags_as_special_tokens
        if not include_tags:
            raise NotImplementedError
        self.include_tags = include_tags

        self.left_tag_to_id = {}
        self.right_tag_to_id = {}

    def generate_mask(self, token_ids, prompt_mask, sample):
        mask = [0] * len(token_ids)
        left_tag, right_tag = sample['output_core_tag_left'], sample[
            'output_core_tag_right']
        if left_tag not in self.left_tag_to_id:
            if left_tag is None:
                left_token_ids = None
            else:
                left_token_ids = self.tokenizer(
                    left_tag,
                    add_special_tokens=False,
                    return_attention_mask=False)['input_ids']
            self.left_tag_to_id[left_tag] = left_token_ids
        else:
            left_token_ids = self.left_tag_to_id[left_tag]
        if right_tag not in self.right_tag_to_id:
            if right_tag is None:
                right_token_ids = None
            else:
                right_token_ids = self.tokenizer(
                    right_tag,
                    add_special_tokens=False,
                    return_attention_mask=False)['input_ids']
            self.right_tag_to_id[right_tag] = right_token_ids
        else:
            right_token_ids = self.right_tag_to_id[right_tag]

        cur_ = 0
        for idx in range(len(token_ids)):
            if prompt_mask[idx] == 1 or token_ids[
                    idx] == self.tokenizer.bos_token_id:
                cur_ = 0
                continue

            if left_token_ids is None:
                match_left = True
            else:
                match_left = True
                try:
                    for offset in range(len(left_token_ids)):
                        if token_ids[idx + offset] != left_token_ids[offset]:
                            match_left = False
                            break
                except IndexError:
                    match_left = False

            if match_left:
                cur_ = 1

            mask[idx] = cur_

            if right_token_ids is None:
                continue

            match_right = True
            try:
                for offset in range(len(right_token_ids)):
                    if token_ids[idx - len(right_token_ids) +
                                 offset] != right_token_ids[offset]:
                        match_right = False
                        break
            except IndexError:
                match_right = False

            if match_right:
                cur_ = 0

        return mask
