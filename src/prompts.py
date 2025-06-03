MULTIPLE_PROMPT_FORCE = 'You are a helpful assistant, below is a query from a user and some relevant contexts. \
Answer the question given the information in those contexts. Your answer should be short and concise and must come from contexts. \
\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:'

SELF_CITATION_PROMPT = """You are a helpful assistant, below is a query from a user, some relevant contexts, and an answer to the query. 
Please cite the top [k] most important contexts that lead to the answer using their indexes, and order these [k] contexts from most important to least important. e.g.,[10]>[32]>[6]>[8]>[25]. ">" means "more important than". Only output these indexes.
\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer: [answer]."""

def wrap_prompt(question, context, split_token = "") -> str:
    assert type(context) == list
    context_str = split_token.join(context)
    input_prompt = MULTIPLE_PROMPT_FORCE.replace('[question]', question).replace('[context]', context_str)
    return input_prompt

def wrap_prompt_self_citation(question, context,answer,k = 5) -> str:

    assert type(context) == list
    context_str = "\n".join(context)

    input_prompt = SELF_CITATION_PROMPT.replace('[question]', question).replace('[context]', context_str).replace('[answer]', answer).replace('[k]', str(k))
    return input_prompt

