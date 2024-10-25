def get_decompose_prompt(
        task,
        history):
    decompose_prompt = f"""Please continue to decompose the provided main question into answerable sub-questions following previously already solved sub-questions. There are two cases as follows:
(1) [Next] If the question requires further decomposition: Identify and output the next logical sub-question that must be addressed in order to progress towards answering the main question.
(2) [Finish] It means the question does not require further decomposition and can be answered as is.

Now Please Complete the Following Task. Please ensure that each sub-question is specific enough to understand in isolation.
Main Question: {task}{history}
Output:"""
    return decompose_prompt


def get_relevance_prompt(
        task,
        history,
        query,
        docs):
    relevance_prompt = f"""Given a sub-question derived from the main question and a document with its title (an entity name), please assess whether the title/entity is relevant with the sub-question based on the title and shown content of the document. Assign one of the following two categories:
(1) [Relevant]: Choose this category if the given document is relevant with the sub-question.
(2) [Irrelevant]: Select this category if the document is irrelevant with the sub-question.

Now Please Complete the Following Task:
Main Question: {task}{history}
Next Sub-Question: {query}
Documents: {docs}
Next Sub-Question: {query}
Output:"""
    return relevance_prompt


def get_solve_prompt(
        task,
        history,
        query,
        docs):
    solve_prompt = f"""Please assess whether the sub-question derived from the main question can be answered using the information from the provided passages. Your evaluation should categorize the sufficiency of the information in the passages with respect to the sub-question. Assign one of the following three categories:
(1) [Unanswerable]: Choose this category if the given passages do not contain information to answer it directly.
(2) [Answerable]: Use this category if one of the given passages contains sufficient information to directly answer the sub-question. Provide a clear and concise answer to the sub-question, and the ID of the the corresponding passage.

Now Please Complete the Following Task:
Main Question: {task}{history}
Next Sub-Question: {query}
Passages: {docs}
Next Sub-Question: {query}
Output:"""
    return solve_prompt

def get_finish_prompt(
        task,
        psgs,
        history):
    finish_prompt = (f"""Answer the question based on the provided passages and decomposed question-answering pairs. Your output should be 'yes/no' or a short entity.

Question: {task}
Passages: {psgs}{history}
Question: {task}
Output:""")
    return finish_prompt
