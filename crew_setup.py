from crewai import Agent, Crew, Task
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente
load_dotenv()

# Verifica se a chave API está disponível
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY não encontrada nas variáveis de ambiente. Verifique seu arquivo .env")

os.environ["OPENAI_API_KEY"] = openai_key

# Define Agents
code_reviewer = Agent(
    name="Code Reviewer",
    role="Senior Developer",
    goal="Review pull requests for code quality, best practices, and readability.",
    backstory="Experienced software engineer with a focus on clean and efficient code.",
    llm=ChatOpenAI(model="gpt-4"),
)

security_analyst = Agent(
    name="Security Analyst",
    role="Security Specialist",
    goal="Identify security vulnerabilities in the code and suggest fixes.",
    backstory="Expert in security best practices and threat mitigation.",
    llm=ChatOpenAI(model="gpt-4"),
)

# Define Tasks
review_task = Task(
    agent=code_reviewer,
    description="Analyze the pull request and provide feedback on code quality.",
    expected_output="Feedback on code quality, best practices, and readability."
)

security_task = Task(
    agent=security_analyst,
    description="Check for security vulnerabilities in the code changes.",
    expected_output="Identification of security vulnerabilities and suggested fixes."
)

# Create the Crew
crew = Crew(
    agents=[code_reviewer, security_analyst],
    tasks=[review_task, security_task],
)

print("Crew criada com sucesso!")
