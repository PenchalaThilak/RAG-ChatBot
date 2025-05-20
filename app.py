!pip install langchain transformers sentence-transformers faiss-cpu pandas numpy torch
import numpy as np
import pandas as pd
dataset = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/ChatBot questions.csv")
dataset
from langchain.docstore.document import Document
# Install necessary libraries
!pip install langchain transformers sentence-transformers faiss-cpu pandas numpy torch langchain-community
# Task 1: Data Loading
# This is a comment indicating the start of the first task, which involves loading data into the program.

from google.colab import drive
# Imports the 'drive' module from the 'google.colab' library, which allows the code to interact with Google Drive for file access.

import pandas as pd
# Imports the pandas library with the alias 'pd'. Pandas is used for handling data in tabular form, like spreadsheets.

from langchain.docstore.document import Document
# Imports the 'Document' class from the LangChain library, which is used to create structured documents with content and metadata.

# Mount Google Drive
# This is a comment indicating the next step, which is connecting to Google Drive to access files.

drive.mount('/content/drive')
# Mounts Google Drive to the '/content/drive' directory in the Google Colab environment. This prompts the user to authenticate and allows access to files stored in Google Drive.

def dataframe_to_documents(df: pd.DataFrame, content_col: str = "answer", metadata_cols: list = ["question"]) -> list[Document]:
    """
    Converts a pandas DataFrame to a list of LangChain Document objects.

    Args:
        df (pd.DataFrame): Input DataFrame containing the dataset.
        content_col (str): Column name for the main content (default: "answer").
        metadata_cols (list): List of column names to include in metadata (default: ["question"]).

    Returns:
        list[Document]: List of LangChain Document objects.
    """
    # Defines a function named 'dataframe_to_documents' that converts a pandas DataFrame into a list of LangChain Document objects.
    # The function takes three parameters:
    # - df: A pandas DataFrame (the input data table).
    # - content_col: The name of the DataFrame column containing the main text content (defaults to "answer").
    # - metadata_cols: A list of column names to include as metadata in each Document (defaults to ["question"]).
    # The function returns a list of LangChain Document objects.
    # The docstring (text in triple quotes) explains the function's purpose, parameters, and return value.

    try:
        # Starts a try-except block to handle potential errors during the conversion process, preventing the program from crashing.

        documents = [
            Document(
                page_content=str(row[content_col]),
                metadata={col: str(row[col]) for col in metadata_cols}
            )
            for _, row in df.iterrows()
        ]
        # Creates a list of Document objects using a list comprehension.
        # - df.iterrows(): Iterates over each row in the DataFrame, where '_' ignores the row index and 'row' contains the row data.
        # - For each row, a Document object is created:
        #   - page_content=str(row[content_col]): Sets the main content of the Document to the value in the column specified by 'content_col' (converted to a string).
        #   - metadata={col: str(row[col]) for col in metadata_cols}: Creates a dictionary for metadata, where each key is a column name from 'metadata_cols', and the value is the corresponding row value (converted to a string).
        # The result is a list of Document objects stored in the 'documents' variable.

        return documents
        # Returns the list of Document objects.

    except Exception as e:
        # Catches any errors that occur during the try block and stores the error message in the variable 'e'.

        print(f"Error converting DataFrame to Documents: {e}")
        # Prints an error message to the console, including the specific error 'e' that occurred.

        return []
        # Returns an empty list if an error occurs, ensuring the function doesn't fail completely.

# Load the dataset
# This is a comment indicating the next step, which is loading the dataset from a file.

file_path = "/content/drive/MyDrive/Colab Notebooks/ChatBot questions.csv"
# Defines the variable 'file_path' with the path to a CSV file located in Google Drive under the "Colab Notebooks" folder.

dataset = pd.read_csv(file_path)
# Uses pandas to read the CSV file at 'file_path' and stores it as a DataFrame in the variable 'dataset'.

documents = dataframe_to_documents(dataset, content_col="answer", metadata_cols=["question"])
# Calls the 'dataframe_to_documents' function, passing:
# - The 'dataset' DataFrame.
# - content_col="answer" to specify that the "answer" column contains the main content.
# - metadata_cols=["question"] to specify that the "question" column should be included as metadata.
# The resulting list of Document objects is stored in the 'documents' variable.

print(f"Loaded {len(documents)} documents")
# Prints the number of Document objects created, using len(documents) to count them.

# Optional: Inspect the first document
# This is a comment indicating an optional step to check the first document in the list.

if documents:
    # Checks if the 'documents' list is not empty to avoid errors when accessing its elements.

    print(f"First document: {documents[0].page_content}, Metadata: {documents[0].metadata}")
    # Prints the content and metadata of the first Document object in the 'documents' list.
    # - documents[0].page_content: The main content (from the "answer" column) of the first document.
    # - documents[0].metadata: The metadata (from the "question" column) of the first document.
# Task 2: RAG Pipeline
# This comment indicates the start of the second task, which focuses on setting up a Retrieval-Augmented Generation (RAG) pipeline for question-answering.

from langchain.vectorstores import FAISS
# Imports the FAISS class from LangChain, which is used to create a vector store for efficient similarity search of text embeddings.

from langchain.chains import RetrievalQA
# Imports the RetrievalQA class from LangChain, which creates a question-answering chain that combines a language model with a retriever.

from langchain.llms import HuggingFacePipeline
# Imports the HuggingFacePipeline class from LangChain, which integrates Hugging Face models into LangChain’s framework.

from langchain.embeddings import HuggingFaceEmbeddings
# Imports the HuggingFaceEmbeddings class from LangChain, used to generate text embeddings (numerical representations) with Hugging Face models.

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# Imports specific classes from the Hugging Face transformers library:
# - AutoTokenizer: For tokenizing text (converting text to numerical tokens for model input).
# - AutoModelForSeq2SeqLM: For loading a sequence-to-sequence language model (used for text generation).
# - pipeline: For creating a pipeline to simplify model inference tasks like text generation.

import torch
# Imports the PyTorch library, which is used to check for GPU availability and handle tensor operations for the model.

class RAGPipeline:
    # Defines a Python class named RAGPipeline to encapsulate the RAG system functionality.

    def __init__(self, documents: list[Document]):
        """
        Initializes the RAG pipeline with documents.

        Args:
            documents (list[Document]): List of LangChain Documents.
        """
        # Defines the constructor method for the RAGPipeline class, which initializes the pipeline.
        # Takes a parameter 'documents', which is a list of LangChain Document objects (from Task 1).
        # The docstring explains the method’s purpose and the expected input.

        self.documents = documents
        # Stores the input list of documents in the instance variable 'self.documents' for use in the pipeline.

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Initializes a HuggingFaceEmbeddings object with the model "all-MiniLM-L6-v2" from sentence-transformers.
        # This model converts text (like document content) into numerical embeddings for similarity search.

        self.llm = self._setup_llm()
        # Calls the internal method '_setup_llm' to configure the language model and stores it in 'self.llm'.

        self.vector_store = None
        # Initializes 'self.vector_store' as None, to be set up later with a FAISS vector store.

        self.qa_chain = None
        # Initializes 'self.qa_chain' as None, to be set up later with a RetrievalQA chain.

        self._setup_pipeline()
        # Calls the internal method '_setup_pipeline' to configure the vector store and question-answering chain.

    def _setup_llm(self):
        """
        Sets up the Hugging Face LLM))? LLM pipeline.

        Returns:
            HuggingFacePipeline: Configured LLM.
        """
        # Defines a private method to set up the Hugging Face language model pipeline.
        # The docstring explains that it returns a configured HuggingFacePipeline object.

        try:
            # Starts a try-except block to handle potential errors during model setup.

            model_name = "google/flan-t5-base"
            # Specifies the name of the model to use: "google/flan-t5-base", a sequence-to-sequence model for text generation.

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Loads the tokenizer for the specified model, which converts text into tokens the model can process.

            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            # Loads the pre-trained sequence-to-sequence model from Hugging Face using the specified model name.

            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                device=0 if torch.cuda.is_available() else -1  # Use GPU if available
            )
            # Creates a Hugging Face pipeline for text-to-text generation:
            # - "text2text-generation": Specifies the task type (generating text from text input).
            # - model: The loaded model.
            # - tokenizer: The loaded tokenizer.
            # - max_length: Limits the generated text to 512 tokens.
            # - device: Uses GPU (device=0) if available (checked with torch.cuda.is_available()), otherwise CPU (device=-1).

            return HuggingFacePipeline(pipeline=pipe)
            # Wraps the Hugging Face pipeline in a LangChain-compatible HuggingFacePipeline object and returns it.

        except Exception as e:
            # Catches any errors that occur during the try block.

            print(f"Error setting up LLM: {e}")
            # Prints an error message with the specific error 'e'.

            return None
            # Returns None if an error occurs, indicating the LLM setup failed.

    def _setup_pipeline(self):
        """
        Sets up the FAISS vector store and RetrievalQA chain.
        """
        # Defines a private method to set up the FAISS vector store and the RetrievalQA chain.
        # The docstring explains the method’s purpose.

        try:
            # Starts a try-except block to handle potential errors during pipeline setup.

            self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
            # Creates a FAISS vector store from the provided documents and embeddings.
            # - self.documents: The list of Document objects.
            # - self.embeddings: The embedding model to convert document text into vectors.
            # The vector store allows for efficient similarity searches to find relevant documents.

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            # Sets up a RetrievalQA chain:
            # - llm: The language model (from _setup_llm).
            # - chain_type="stuff": A method that combines retrieved documents into a single prompt for the LLM.
            # - retriever: Uses the FAISS vector store as a retriever, configured to return the top 3 most relevant documents (k=3).
            # - return_source_documents=True: Ensures the source documents used for the answer are included in the response.

        except Exception as e:
            # Catches any errors that occur during the try block.

            print(f"Error setting up pipeline: {e}")
            # Prints an error message with the specific error 'e'.

    def query(self, question: str) -> dict:
        """
        Queries the RAG pipeline with a question.

        Args:
            question (str): User question.

        Returns:
            dict: Response containing answer and source documents.
        """
        # Defines a method to query the RAG pipeline with a user question.
        # Takes a string 'question' as input and returns a dictionary with the answer and source documents.
        # The docstring explains the method’s purpose, input, and output.

        if not self.qa_chain:
            # Checks if the RetrievalQA chain (self.qa_chain) is initialized (not None).

            return {"error": "Pipeline not initialized"}
            # Returns an error message in a dictionary if the pipeline isn’t set up.

        try:
            # Starts a try-except block to handle potential errors during the query.

            return self.qa_chain.invoke({"query": question})
            # Calls the RetrievalQA chain with the user’s question (passed as a dictionary with key "query").
            # The chain retrieves relevant documents, generates an answer using the LLM, and returns a dictionary with the answer and source documents.

        except Exception as e:
            # Catches any errors that occur during the query.

            return {"error": f"Query failed: {e}"}
            # Returns a dictionary with an error message if the query fails.

# Initialize RAG pipeline
# This comment indicates the step of creating an instance of the RAGPipeline class.

rag = RAGPipeline(documents)
# Creates an instance of the RAGPipeline class, passing the 'documents' list (from Task 1) to the constructor.
# This initializes the embeddings, language model, vector store, and QA chain.

print("RAG pipeline initialized")
# Prints a confirmation message to indicate that the RAG pipeline has been successfully set up.
# Task 3: Chatbot
# This comment indicates the start of the third task, which focuses on creating and testing a chatbot using the RAG pipeline.

class Chatbot:
    # Defines a Python class named `Chatbot` to encapsulate the chatbot functionality.

    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initializes the Chatbot with a RAG pipeline.

        Args:
            rag_pipeline (RAGPipeline): Initialized RAG pipeline.
        """
        # Defines the constructor method for the `Chatbot` class.
        # Takes a parameter `rag_pipeline`, which is an initialized `RAGPipeline` object (from Task 2).
        # The docstring explains the method’s purpose and the expected input.

        self.rag = rag_pipeline
        # Stores the provided `RAGPipeline` object in the instance variable `self.rag` for use in answering questions.

    def ask(self, question: str) -> str:
        """
        Processes a user question and returns the answer.

        Args:
            question (str): User question.

        Returns:
            str: Formatted response.
        """
        # Defines a method named `ask` to process a user’s question and return a formatted response.
        # Takes a string `question` as input and returns a string containing the answer and sources.
        # The docstring explains the method’s purpose, input, and output.

        response = self.rag.query(question)
        # Calls the `query` method of the RAG pipeline (from Task 2) with the user’s question and stores the result in `response`.
        # The response is a dictionary containing the answer and source documents (or an error message if the query failed).

        if "error" in response:
            # Checks if the response dictionary contains an "error" key, indicating a problem with the query.

            return f"Error: {response['error']}"
            # Returns a formatted error message if an error occurred, using the error details from the response.

        answer = response["result"]
        # Extracts the answer from the response dictionary (the key "result" holds the generated answer from the RAG pipeline).

        sources = [doc.page_content for doc in response["source_documents"]]
        # Creates a list of the content (page_content) from each source document in the response.
        # `response["source_documents"]` contains the relevant documents retrieved by the RAG pipeline.

        return f"Answer: {answer}\nSources: {sources}"
        # Returns a formatted string combining the answer and the list of source document contents.

# Initialize Chatbot
# This comment indicates the step of creating an instance of the `Chatbot` class.

chatbot = Chatbot(rag)
# Creates an instance of the `Chatbot` class, passing the `rag` object (the initialized `RAGPipeline` from Task 2) to the constructor.

# Test the Chatbot and Save Responses
# This comment indicates the step of testing the chatbot with sample questions and saving the responses.

test_questions = [
    "What is Artificial Intelligence?",
    "What is RAG?",
    "What is a Chatbot?"
]
# Defines a list of test questions to evaluate the chatbot’s performance.

responses = []
# Initializes an empty list called `responses` to store the formatted question-and-answer pairs.

for question in test_questions:
    # Starts a loop to iterate over each question in the `test_questions` list.

    response = chatbot.ask(question)
    # Calls the `ask` method of the chatbot with the current question and stores the formatted response (answer and sources) in `response`.

    responses.append(f"Question: {question}\n{response}\n")
    # Appends a formatted string to the `responses` list, combining the question, the response, and an extra newline for readability.

    print(f"Question: {question}\n{response}\n")
    # Prints the question and its response to the console for immediate feedback during testing.

# Save sample responses to a file
# This comment indicates the step of saving the responses to a file.

with open("/content/sample_responses.txt", "w") as f:
    # Opens a file named "sample_responses.txt" in the "/content" directory in write mode ("w").
    # The `with` statement ensures the file is properly closed after writing.
    # The file object is assigned to the variable `f`.

    f.write("\n".join(responses))
    # Writes all responses to the file, joining them with newline characters (`\n`) to separate each response.

# Download the responses file
# This comment indicates the step of downloading the created file in Google Colab.

from google.colab import files
# Imports the `files` module from the `google.colab` library, which provides functionality to download files from the Colab environment.

files.download("/content/sample_responses.txt")
# Triggers the download of the "sample_responses.txt" file to the user’s local machine.


!pip install gradio
# Task 4: Gradio UI
# This comment indicates the start of the fourth task, which focuses on creating a user interface for the chatbot using Gradio.

import gradio as gr
# Imports the Gradio library with the alias `gr`. Gradio is used to create simple web-based interfaces for Python functions.

def chatbot_interface(question: str) -> str:
    """
    Processes a question through the Chatbot and returns the response for Gradio.

    Args:
        question (str): User input question.

    Returns:
        str: Chatbot response with answer and sources.
    """
    # Defines a function named `chatbot_interface` to process user questions for the Gradio interface.
    # Takes a string `question` as input and returns a string response.
    # The docstring explains the function’s purpose, input, and output.

    if not question.strip():
        # Checks if the input `question` is empty or contains only whitespace (after stripping whitespace).
        # `question.strip()` removes leading and trailing whitespace from the input string.

        return "Please enter a question."
        # Returns an error message if the question is empty, prompting the user to provide a valid input.

    return chatbot.ask(question)
    # Calls the `ask` method of the `chatbot` object (from Task 3) with the user’s question and returns the formatted response (answer and sources).

# Create Gradio interface
# This comment indicates the step of setting up the Gradio interface.

interface = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(label="Enter your question", placeholder="e.g., What is Artificial Intelligence?"),
    outputs=gr.Textbox(label="Response"),
    title="RAG Chatbot",
    description="Ask questions about AI, chatbots, RAG",
    theme="default"
)
# Creates a Gradio interface object and assigns it to the variable `interface`.
# Parameters:
# - fn=chatbot_interface: Specifies the function (`chatbot_interface`) that processes user inputs.
# - inputs=gr.Textbox(...): Defines the input component as a textbox with a label and placeholder text to guide the user.
# - outputs=gr.Textbox(label="Response"): Defines the output component as a textbox labeled "Response" to display the chatbot’s answer.
# - title="RAG Chatbot": Sets the title of the web interface.
# - description="Ask questions about AI, chatbots, RAG": Provides a description of the interface’s purpose.
# - theme="default": Uses the default Gradio theme for the interface’s appearance.

# Launch the interface
# This comment indicates the step of launching the web interface.

interface.launch(share=True)
# Launches the Gradio interface, making it accessible via a web browser.
# The `share=True` parameter creates a public URL for the interface, hosted by Gradio’s servers, allowing external access.
# In Google Colab, this generates a link that users can click to interact with the chatbot.












