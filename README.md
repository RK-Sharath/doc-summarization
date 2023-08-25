# Summarize the documents using Watsonx LLMs

## Instructions to set it up on the local environment

* Clone the Git repo and navigate to the content generation folder in the terminal.
* Run the command streamlit run /.../doc-summarize.py
* You can access the application at http://localhost:8501/
* Update the credentials API Key & URL from BAM/Workbench in the UI and you are all set to start prompting.
* All the hyperparameters are configurable. Play around with different combinations to get the desired results.
* Choose google/flan-t5-xxl as the Watsonx LLM model to get started as it gives good results.

### Sample Prompts

* prompt_template = """Write a summary of the document:

{text}

Extract highlights from the text in 250 words:

{text}

Create the summary of the highlighted text in 150 words:

{text}

Refine the summary if possible

* prompt_template = """Write a summary of the document:

{text}

Summary:

"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary\n"
    "If the context isn't useful, return the original summary."
)
