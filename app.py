
from fastapi import FastAPI, HTTPException
import openai
from agent3 import graph_app
#from demo import graph_app
from prompt import travel_visa,study_visa,work_visa
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

def get_roadmap_from_type(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "You are Paddi AI, a visa advisor specializing in personalized roadmaps for visa applications"},
        {"role": "user", "content": prompt}
    ],
    temperature=0.6 
    )

    # Extract and return the model's response
    return response.choices[0].message.content

# Define API endpoint
@app.post("/generate_roadmap")
async def generate_visa_roadmap(questionnaire:str, roadmap_type:str):
    if roadmap_type.lower() == "immigration visa":
        try:
            result = graph_app.invoke({"questionnaire": questionnaire})
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) 
           
    elif roadmap_type.lower() == "study visa":
        prompt = study_visa(questionnaire)
        return get_roadmap_from_type(prompt)
    
    elif roadmap_type.lower() == "travel visa":
        prompt = travel_visa(questionnaire)
        return get_roadmap_from_type(prompt)

    elif roadmap_type.lower() == "work visa":
        prompt = work_visa(questionnaire)
        return get_roadmap_from_type(prompt)

    else:
        return "Invalid type."


# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host= "0.0.0.0", port=8000)
