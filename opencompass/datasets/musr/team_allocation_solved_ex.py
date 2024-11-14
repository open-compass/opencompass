# flake8: noqa: E501
story = '''
In the quaint community of Midvale, the local school stood as a beacon of enlightenment, nurturing the minds of the next generation. The teachers, the lifeblood of this institution, were tasked with the noble duty of education, while the unsung heroes—the maintenance crew—ensured the smooth functioning of the school's infrastructure. Amidst this, three town residents, Angela, Greg, and Travis, found themselves at a juncture of life where they were presented with the opportunity to serve in one of these crucial roles. The challenge now lay in the hands of the manager, who had to assign them to either teaching or maintenance, a decision that would set the course for their contributions to the school.

Angela was a fiercely independent woman, beset with a unique set of strengths and weaknesses. She was a woman of very few words, often finding it hard to articulate her thoughts and explain things clearly. Venturing into education seemed a maze with her apathetic attitude towards learning. She was also seen to be disinterested in reading and the literary field as a whole. This was a juxtaposition to her inability to contribute to maintenance duties because of her fear of tools and machinery, a sinister remnant of a past accident that still haunted her. The basic handyman skills, which most locals learned growing up, were also absent from her repertoire.

Angela's interactions with Greg and Travis further complicated the equation. On one hand, Greg and Angela had a habit of arguing constantly over trivial matters, which once culminated in their failure to complete a shared basic training exercise adequately. On the other hand, Angela and Travis simply had nothing in common. Their conversations were often fraught with awkward silences, indicative of their lack of shared interests. This lack of coordination was epitomized during a recent team-building exercise when their team finished last.

Greg was the blue-collar type with a broad frame and muscular build. He had a work ethic that never shied away from toiling through the day to get things done. Growing up, he often helped his father with simple home repairs and minor renovations, learning the ropes of basic handiwork. Additionally, Greg had fortified his skills while refurbishing an old shed with Travis, a testament to their compatible personalities. However, his dislike for education was well known throughout town, further amplified by his lack of patience, especially with children.

Travis, the third cog in the wheel, was a man of many peculiarities. His stage fright was almost legendary and made it nearly impossible for him to stand in front of a crowd. Often, the mere thought of it could unnerve him. His physical constitution was lightweight and fragile, and long hours of manual labor made him weary. He also had a revulsion towards dirt that he complained about at every opportune moment. Like the others, studying did not appeal to him much, so much so that he had stopped reading completely after leaving school prematurely.

The manager understood well that a team’s success depends heavily on the contribution and compatibility of each member. He observed, analyzed, and considered. Now, it was up to him to assign roles to Angela, Greg, and Travis. The school needed educators and maintenance staff, and each had to play their part perfectly.

Given the story, how would you uniquely allocate each person to make sure both tasks are accomplished efficiently?

Pick one of the following choices:
1 - Teaching: Travis, Maintenance: Angela and Greg
2 - Teaching: Greg, Maintenance: Angela and Travis
3 - Teaching: Angela, Maintenance: Greg and Travis

You must pick one option. The story should allow you to determine how good each person is at a skill. Roughly, each person is either great, acceptable, or bad at a task. We want to find an optimal assignment of people to tasks that uses their skills as well as possible. In addition, one task will have to have two people assigned to it. The effectiveness of their teamwork (great team, acceptable team, or bad team) also impacts the overall quality of the assignment.

When two people need to work on a task and one is bad at it, they don’t necessarily benefit from the other person being good, unless they work well together.

With different strengths, weaknesses, and interpersonal dynamics at play, you should allocate your team to find the single assignment to ensure that the tasks overall are completed as effectively as possible.

 Explain your reasoning step by step before you answer. Finally, the last thing you generate should be "ANSWER: (your answer here, including the choice number)"
'''.strip()

reasoning = '''
Let's solve this by thinking step-by-step. First, we will figure out each person's skill level for each task. Then, we can measure how well they all work together in pairs. From this, we can find the most efficient assignment that maximizes the scores.

Let's start with Angela. Angela can't articulate her thoughts, and she seems unprepared for teaching. So, let's assume her skill level is 1 for teaching. She also is bad at maintenance due to her fear of maintenance. So, let's assume her skill level is 1 for maintenance as well.

Now, let's look at Greg. Greg has a dislike for education and a lack of patience, so let's assume his skill level for maintenance is 1. However, Greg has helped with home repairs and minor renovations, so let's assume his maintenance skill level is 2.

Finally, let's look at Travis. Travis has extreme stage fright, which will make it difficult to teach, so let's assume his teaching skill level is 1. He also has a lightweight and fragile frame as well as hates dirt, so let's assume his maintenance skill level is 1.

Now, let's look at the relationships and how people work together.

Angela and Greg do not get along; they are constantly arguing, so let's assume their ability to work together is 1.

Angela and Travis aren't much better. They both have nothing in common, and they couldn't do a team-building exercise previously, so let's assume their ability to work together is 1.

Finally, Greg and Travis have worked together, and their personalities seem to meld, so let's assume they work well together with a score of 3.

Let's summarize and figure out the best assignment.

Angela is bad at teaching. (1)
Angela is bad at maintenance. (1)
Angela does not work well with Greg. (1)
Angela does not work well with Travis. (1)
Greg is bad at teaching. (1)
Greg is okay with maintenance. (2)
Greg and Travis work well together. (3)
Travis is bad at teaching. (1)
Travis is bad at maintenance. (1)

Now, let's find the best assignment.

Option 1: Travis as a teacher (1) + Angela working in maintenance (1) + Greg working in maintenance (2) + Angela and Greg work badly together (1) = 5
Option 2: Greg as a teacher (1) + Angela working in maintenance (1) + Travis working in maintenance (1) + Angela and Travis work badly together (1) = 4
Option 3: Angela as a teacher (1) + Greg working in maintenance (2) + Travis working in maintenance (1) + Greg and Travis work well together (3) = 7

So, from this, we can see Option 3 has the maximum score.

ANSWER: 3

'''.strip()

team_allocation_solved_ex = f'{story}\n\n{reasoning}'
