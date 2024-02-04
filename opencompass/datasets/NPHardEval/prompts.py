# Overall fewshot prompts
FEW_SHOT_SELF = 'Please refer to a few examples of this problem and the corresponding reasoning process. The examples are:'
FEW_SHOT_OTHERS = 'Please refer to a few examples of another problem and the corresponding reasoning process. The problem is {initial_question}. {output_content}. The examples are:'

# P problems
sppPrompts = {
    'Intro': 'The Shortest Path Problem (SPP) involves finding the shortest path between two nodes in a weighted graph.',
    'Initial_question': "You need to find the shortest path between node {start_node} and node {end_node} in a graph. The graph's edges and their weights are given.",
    'Output_content': 'Please provide the shortest path from {start_node} to {end_node} and its total distance. Offer a concise step-by-step explanation of your reasoning process. Aim for brevity and clarity in your response.',
    'Output_format': "Your output should be enclosed within <root></root> tags. Include your reasoning in <reasoning></reasoning> tags and the final path and total distance in <final_answer></final_answer> tags, like <final_answer>{'Path': 'START->...->END', 'TotalDistance': 'INT_TOTAL_DISTANCE'}</final_answer>.",
    'Few_shot_self': FEW_SHOT_SELF,
    'Few_shot_others': FEW_SHOT_OTHERS
}

mfpPrompts = {
    'Intro': 'The Maximum Flow Problem (MFP) seeks to find the maximum possible flow from a source node to a sink node in a flow network, subject to capacity constraints on the edges.',
    'Initial_question': 'Determine the maximum flow from the source node {source_node} to the sink node {sink_node} in the given flow network. The capacities of the edges are provided.',
    'Output_content': 'Please indicate the maximum flow value and the flow for each edge. Provide a brief explanation of your methodology. Keep your response concise and focused.',
    'Output_format': "Enclose your output within <root></root> tags. Present your reasoning in <reasoning></reasoning> tags and the final maximum flow and edge flows in <final_answer></final_answer> tags, like <final_answer>{'MaxFlow': 'MAX_FLOW_VALUE', 'Flows': {'NODE_1->NODE_2': FLOW, ...}}</final_answer>.",
    'Few_shot_self': FEW_SHOT_SELF,
    'Few_shot_others': FEW_SHOT_OTHERS
}

bspPrompts = {
    'Intro': 'The Binary Search Problem (BSP) deals with finding the position of a target value within a sorted array using a binary search algorithm, which efficiently narrows down the search range.',
    'Initial_question': 'Find the position of the target value {target_value} in the sorted array. The index begins with 0. The array elements are provided.',
    'Output_content': 'Please identify the position of the target value in the array. Offer a brief, step-by-step account of your search process. Aim for conciseness in your response.',
    'Output_format': "Your output should be enclosed in <root></root> tags. Include your search process in <reasoning></reasoning> tags and the final position of the target value in <final_answer></final_answer> tags, like <final_answer>{'Position': 'TARGET_POSITION'}</final_answer>.",
    'Few_shot_self': FEW_SHOT_SELF,
    'Few_shot_others': FEW_SHOT_OTHERS
}

edpPrompts = {
    'Intro': 'The Edit Distance Problem (EDP) involves finding the minimum number of operations required to transform one string into another, where each operation is either an insertion, deletion, or substitution of a single character.',
    'Initial_question': 'Find the minimum number of operations required to transform the first string {string_a} into the second string {string_b}. The operations are insertion, deletion, and substitution of a single character, each requiring 1 edit operation.',
    'Output_content': 'Please provide the minimum number of operations required to transform the first string into the second string. Offer a brief explanation of your methodology. Keep your response concise and focused.',
    'Output_format': "Enclose your output within <root></root> tags. Present your reasoning in <reasoning></reasoning> tags and the final minimum number of operations in <final_answer></final_answer> tags, like <final_answer>{'Operations': 'MINIMUM_NUMBER_OF_OPERATIONS'}</final_answer>.",
    'Few_shot_self': FEW_SHOT_SELF,
    'Few_shot_others': FEW_SHOT_OTHERS
}

# NP-complete problems
tsp_dPrompts = {
    'Intro': 'The Traveling Salesman Problem (Decision Version, TSP-D) focuses on determining if a salesman can complete a route, visiting each city at least once, with the total travel distance being less than a specified value.',
    'Initial_question': "Check if it's possible for a salesman to visit each of the {total_cities} cities at least once and return to the starting city with the total distance less than {distance_limit}. The distances between each pair of cities are given.",
    'Output_content': 'Provide a yes or no answer, with a succinct explanation of your decision process. Focus on clarity and brevity in your response.',
    'Output_format': "Enclose your output in <root></root> tags. Present your reasoning in <reasoning></reasoning> tags and the final yes/no answer in <final_answer></final_answer> tags, like <final_answer>{'Feasible': 'YES_OR_NO'}</final_answer>.",
    'Few_shot_self': FEW_SHOT_SELF,
    'Few_shot_others': FEW_SHOT_OTHERS
}

gcp_dPrompts = {
    'Intro': 'The Graph Coloring Problem (Decision Version, GCP-D) involves determining if it is possible to color the vertices of a graph using a given number of colors, ensuring no two adjacent vertices have the same color.',
    'Initial_question': 'Find out if the vertices of a graph with {total_vertices} vertices can be colored using only {number_of_colors} colors, such that no adjacent vertices share the same color.',
    'Output_content': 'Provide a yes or no answer, along with a concise explanation of your reasoning. Keep your explanation focused and brief.',
    'Output_format': "Enclose your output in <root></root> tags. Include your reasoning in <reasoning></reasoning> tags and the final yes/no answer in <final_answer></final_answer> tags, like <final_answer>{'Feasible': 'YES_OR_NO'}</final_answer>.",
    'Few_shot_self': FEW_SHOT_SELF,
    'Few_shot_others': FEW_SHOT_OTHERS
}

kspPrompts = {
    'Intro': 'The 0-1 Knapsack Problem (KSP) asks whether a subset of items, each with a given weight and value, can be chosen to fit into a knapsack of fixed capacity, maximizing the total value without exceeding the capacity.',
    'Initial_question': 'Determine if a subset of items can be selected to fit into a knapsack with a capacity of {knapsack_capacity}, maximizing value without exceeding the capacity. Item weights and values are provided.',
    'Output_content': 'Indicate if an optimal subset exists and its total value. Offer a concise explanation of your selection process. Aim for clarity and brevity in your response.',
    'Output_format': "Your output should be enclosed within <root></root> tags. Include your selection process in <reasoning></reasoning> tags and the final decision and total value in <final_answer></final_answer> tags, like <final_answer>{'Feasible': 'YES_OR_NO', 'TotalValue': 'TOTAL_VALUE', 'SelectedItemIds': [0, 1]}</final_answer>.",
    'Few_shot_self': FEW_SHOT_SELF,
    'Few_shot_others': FEW_SHOT_OTHERS
}

# NP-hard problems
tspPrompts = {
    'Intro': 'The traveling salesman problem (TSP) is a classic optimization problem that aims to find the shortest possible route that visits a set of cities, with each city being visited exactly once and the route returning to the original city.',
    'Initial_question': 'You must find the shortest path that visits all {total_cities} cities, labelled from 1 to {total_cities}. The distances between each pair of cities are provided.',
    'Output_content': 'Please list each city in the order they are visited. Provide the total distance of the trip. You should also provide very short step by step reasoning. Do not use multiple lines and try your best to save output tokens.',
    'Output_format': "Your output should contain two parts enclosed by <root></root>. First, your step by step reasoning like <reasoning>The reasoning process</reasoning>. Second, the final output of the result path and total distance wrapped by final_answer tag, like <final_answer>{'Path': '0->1->2->...->N->0', 'TotalDistance': 'INT_TOTAL_DISTANCE'}</final_answer>",
    'Few_shot_self': FEW_SHOT_SELF,
    'Few_shot_others': FEW_SHOT_OTHERS
}

gcpPrompts = {
    'Intro': 'Graph coloring refers to the problem of coloring vertices of a graph in such a way that no two adjacent vertices have the same color. ',
    'Initial_question': 'There are {max_vertices} vertices 1 to {max_vertices} in a graph. You may use {max_colors} colors with alphabats from A, B, C,... to color the graph.',
    'Output_content': "Please label every vertex, even if it is disconnected from the rest of the graph. Please provide each vertex's color. Do not skip any vertices. You should also provide very short step by step reasoning. Do not use multiple lines and try your best to save output tokens.",
    'Output_format': "Your output should contain two parts enclosed by <root></root>. First, your step by step reasoning wrapped by <reasoning></reasoning>. Second, the final output of all vertex numbers and their associated colors, wrapped by final_answer tag, like <final_answer>{0:'COLOR_1', 1:'COLOR_2', ...}</final_answer>.",
    'Few_shot_self': FEW_SHOT_SELF,
    'Few_shot_others': FEW_SHOT_OTHERS
}

mspPrompts = {
    'Intro': 'The meeting scheduling problem (MSP) is a type of constraint satisfaction problem where the goal is to find a suitable time slot for a meeting that all participants can attend without conflicts in their schedules.',
    'Initial_question': "There are {total_participants} participants with their available time slots. There are {total_timeslots} consecutive non-overlapping time slots. Let's assume all meetings has duration of 1.",
    'Output_content': 'Please provide a time slot where all participants can attend the meeting. You should also provide very short step by step reasoning. Do not use multiple lines and try your best to save output tokens.',
    'Output_format': 'Your output should contain two parts enclosed by <root></root>. First, your step by step reasoning wrapped by <reasoning></reasoning>. Second, the final output of meeting numbers followed by a list of slots, like <final_answer>{0:[1,2], 1:[4], ...}</final_answer>.',
    'Few_shot_self': FEW_SHOT_SELF,
    'Few_shot_others': FEW_SHOT_OTHERS
}
