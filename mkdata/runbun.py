import dspy
import openai

tiny = dspy.OpenAI(
    model="tinyllama:latest",
    model_type="chat",
    max_tokens=300,
    api_base="http://192.168.86.33:11434/v1/",
    api_key="FAKE_API_KEY",
)
dspy.configure(lm=tiny)


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    question = dspy.InputField()
    query = dspy.OutputField()


if __name__ == "__main__":
    get_answer = dspy.Predict(GenerateSearchQuery)
    try:
        response = get_answer(question="what is the meaning of life?")
        print(response.query)
    except openai.NotFoundError as e:
        print(e.response.status_code)
        print(e.response.request)
        print(e.response.content)
    except openai.BadRequestError as e:
        print(e.response.request.content)

        print(e.response.request.url)
