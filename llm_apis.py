from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import logging

app = FastAPI()

# Allow CORS for all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(filename='log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

class GenerateRequest(BaseModel):
    message: str
    model: str = 'gemini-1.5-flash'

@app.post("/api/generate")
async def generate_text(request: GenerateRequest):
    try:
        user_message = request.message
        model_name = request.model

        if not user_message:
            raise HTTPException(status_code=400, detail="No message provided")

        logging.info(f"Received request: model={model_name}, message='{user_message}'")

        headers = {
            'Content-Type': 'application/json'
        }

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"

        request_body = {
            "contents": [{
                "parts": [{"text": user_message}]
            }]
        }

        response = requests.post(api_url, headers=headers, json=request_body)

        response.raise_for_status()
        response_data = response.json()
        logging.info(f"Full API response: {response_data}")

        # Extract the response text from the correct fields
        if 'candidates' not in response_data or not response_data['candidates']:
            raise HTTPException(status_code=500, detail="Invalid response format: 'candidates'")

        response_text = response_data['candidates'][0]['content']['parts'][0]['text']
        logging.info(f"API response: {response_text}")
        return {"response": response_text}

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"API request failed: {e}")
    except (KeyError, IndexError) as e:
        logging.error(f"Invalid response format: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Invalid response format: {e}")
    except ValueError as e:
        logging.error(f"Value Error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
