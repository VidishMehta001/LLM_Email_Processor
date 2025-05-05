# Configuration and imports
import os
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Sample email dataset
sample_emails = [
    {
        "id": "001",
        "from": "angry.customer@example.com",
        "subject": "Broken product received",
        "body": "I received my order #12345 yesterday but it arrived completely damaged. This is unacceptable and I demand a refund immediately. This is the worst customer service I've experienced.",
        "timestamp": "2024-03-15T10:30:00Z"
    },
    {
        "id": "002",
        "from": "curious.shopper@example.com",
        "subject": "Question about product specifications",
        "body": "Hi, I'm interested in buying your premium package but I couldn't find information about whether it's compatible with Mac OS. Could you please clarify this? Thanks!",
        "timestamp": "2024-03-15T11:45:00Z"
    },
    {
        "id": "003",
        "from": "happy.user@example.com",
        "subject": "Amazing customer support",
        "body": "I just wanted to say thank you for the excellent support I received from Sarah on your team. She went above and beyond to help resolve my issue. Keep up the great work!",
        "timestamp": "2024-03-15T13:15:00Z"
    },
    {
        "id": "004",
        "from": "tech.user@example.com",
        "subject": "Need help with installation",
        "body": "I've been trying to install the software for the past hour but keep getting error code 5123. I've already tried restarting my computer and clearing the cache. Please help!",
        "timestamp": "2024-03-15T14:20:00Z"
    },
    {
        "id": "005",
        "from": "business.client@example.com",
        "subject": "Partnership opportunity",
        "body": "Our company is interested in exploring potential partnership opportunities with your organization. Would it be possible to schedule a call next week to discuss this further?",
        "timestamp": "2024-03-15T15:00:00Z"
    }
]

class EmailProcessor:
    def __init__(self):
        """Initialize the email processor with OpenAI API key."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Define valid categories
        self.valid_categories = {
            "complaint", "inquiry", "feedback",
            "support_request", "other"
        }

    def classify_email(self, email: Dict) -> Optional[str]:
        """
        Classify an email using LLM.
        Returns the classification category or None if classification fails.
        """
        try:
            #Query few shot samples
            with open("synthetic_samples/generated_samples.json", 'r', encoding='utf-8') as f:
                res = str(json.load(f))

            # Initial prompt
            prompt = f"""
            Imagine you are a email classifier. Please classify the email below according to the provided categories.

            Provided categories: {list(self.valid_categories)}
    
            Email: {email}

            Please find several examples with category provided in the "category" key: {res}

            Provide a JSON format (dictinonary) with key as classification and value as the valid category classified to. Please only provide this and no other writeups.
            If the email is not from a customer, it belong to other category - for instance if the email is from internal teams, external collaborators or partnerships.
            """

            # Saving prompt information to classify_prompt_v1
            # with open ("classify_prompt_v5.txt", 'w') as f:
            #     f.write(prompt)

            # Auto versioning up the classify prompt
            save_file_version(filename=f"classify_prompt_{str(email['id'])}", folder="Prompts", filedata=prompt)

            # Generating the response - api call to chat gpt open ai
            response = self.client.chat.completions.create(
                messages=[
                    {"role":"system", "content": "Imagine you are an expert at classifying emails"},
                    {"role": "user", "content": prompt}
                ],
                model="gpt-4o-mini",
                response_format={"type":"json_object"}
            )

            # Json object documenting the response from the LLM models - in terms of classification
            response_json = response.choices[0].message.content
            # Load the response in json format
            response_json = json.loads(response_json)
            # Get the classification results
            classification = response_json['classification']
            # Log the classification results
            logger.info(f"Final classification for the email: {classification}")
            return classification
        except Exception as e:
            logger.error(f"Error classifying the email: {str(e)}")

    def generate_response(self, email: Dict, classification: str) -> Optional[str]:
        """
        Generate an automated response based on email classification.
        """
        try:
            # Initial prompt
            prompt = f"""
            Generate a response to the following email based on the classification: {classification}

            Email: {email}
            
            Response should comprise the text body of the email. 
            Please provide the response in a json format (dictionary) with key 'email_body': response.

            if the requestor is a customer, please address them as Dear Valued Customer. In the sign off, mention name as Vidish Mehta, company as Cornell University, Contact Information as +1 123 456 789 and Customer Support team.
            if the requestor is a internal team member or an external collaborator, please address them as Dear Sir/Madam. In the sign off, mention name as Vidish Mehta, company as Cornell University, Contact Information as +1 123 456 789 and Business Development team.
            """

            # Log the response prompt version 1
            # with open("response_prompt_v3.txt", 'w') as f:
            #     f.write(prompt)

            # Auto versioning up the response prompt
            save_file_version(filename=f"response_prompt_{str(email['id'])}", folder="Prompts", filedata=prompt)

            # Generating the response - api call to chat gpt
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content":"Imagine you are an expert at replying to customer support emails"},
                    {"role": "user", "content": prompt}
                    ],
                response_format={"type":"json_object"}
            )

            # Parse the response
            response_json = response.choices[0].message.content
            # Load the response into json
            response_json = json.loads(response_json)
            # Parse the response text
            response_text = response_json['email_body']
            # Remove line breaks.
            response_text = response_text.replace("\n\n", " ")
            response_text = response_text.replace("\n", " ")

            # clean response text
            response_text = " ".join(response_text.split())

            # Log the response provided by the LLM
            logger.info(response_text)
            return response_text
        
        except Exception as e:
            logger.error(f"Error generating response to the email: {str(e)}")
            return None


class EmailAutomationSystem:
    def __init__(self, processor: EmailProcessor):
        """Initialize the automation system with an EmailProcessor."""
        self.processor = processor
        self.response_handlers = {
            "complaint": self._handle_complaint,
            "inquiry": self._handle_inquiry,
            "feedback": self._handle_feedback,
            "support_request": self._handle_support_request,
            "other": self._handle_other
        }

    def process_email(self, email: Dict) -> Dict:
        """
        Process a single email through the complete pipeline.
        Returns a dictionary with the processing results.
        """

        try:
            # Step 1: Classifying the email
            classification = self.processor.classify_email(email=email)
            if not classification:
                raise ValueError("Error classifying the email")
            
            # Step 2: Generating the response
            response = self.processor.generate_response(email=email, classification=classification)
            if not response:
                raise ValueError("Error generating the response to the email")
            
            # Step 3: Query the appropriate handler
            handler = self.response_handlers.get(classification)
            if not handler:
                raise ValueError("Error querying the appropriate handler")

            # Step 4: Run the handler 
            res = handler(email=email)

            # Step 5: Log the results 
            #  print(df[["email_id", "success", "classification", "response_sent"]])
            results = {
                "email_id": email['id'],
                "success": "Yes",
                "classification": classification,
                "response_sent": response
            }

            return results
        
        except Exception as e:
            logger.error(f"Error processing emails: {str(e)}") 
            results = {
                "email_id": email['id'],
                "success": "No",
                "classification": str(e),
                "response": str(e)
            }
            return results

    def _handle_complaint(self, email: Dict):
        """
        Handle complaint emails.
        """
        try:
            # Generate the response
            response = self.processor.generate_response(email=email, classification="complaint")
            if not response:
                raise ValueError("Failed to generate response")
            
            # Send the resposne
            send_complaint_response(email_id=email['id'], response=response)

            # Create urgent ticket 
            create_urgent_ticket(email_id=email['id'], category="complaint", context=email['body'])

            return None
        
        except Exception as e:
            logger.error(f"Error in processing complaints: {str(e)}")
            return None

    def _handle_inquiry(self, email: Dict):
        """
        Handle inquiry emails.
        """
        try:
            # Generate the response
            response = self.processor.generate_response(email=email, classification="inquiry")
            if not response:
                raise ValueError("Failed to generate response")
            
            # Send the standard resposne
            send_standard_response(email_id=email['id'], response=response)

            # Create support ticket
            create_support_ticket(email_id=email['id'], context=email['body'])

            return None
        
        except Exception as e:
            logger.error(f"Error in processing inquiry: {str(e)}")
            return None

    def _handle_feedback(self, email: Dict):
        """
        Handle feedback emails.
        """
        try:
            # Generate the response
            response = self.processor.generate_response(email=email, classification="feedback")
            if not response:
                raise ValueError("Failed to generate response")
            
            # Send the resposne
            send_standard_response(email_id=email['id'], response=response)

            # Log customer feedback
            log_customer_feedback(email_id=email['id'], feedback=email['body'])

            return None
        
        except Exception as e:
            logger.error(f"Error in processing feedback: {str(e)}")
            return None

    def _handle_support_request(self, email: Dict):
        """
        Handle support request emails.
        """
        try:
            # Generate the response
            response = self.processor.generate_response(email=email, classification="support_request")
            if not response:
                raise ValueError("Failed to generate response")
            
            # Send the resposne
            send_standard_response(email_id=email['id'], response=response)

            # Create a support ticket
            create_support_ticket(email_id=email['id'], context=email['body'])
            
            return None
        
        except Exception as e:
            logger.error(f"Error in processing support request: {str(e)}")
            return None


    def _handle_other(self, email: Dict):
        """
        Handle other category emails.
        """
        try:
            # Generate the response
            response = self.processor.generate_response(email=email, classification="other")
            if not response:
                raise ValueError("Failed to generate response")
            
            # Send the resposne
            send_standard_response(email_id=email['id'], response=response)
    
            return None
        
        except Exception as e:
            logger.error(f"Error in processing support request: {str(e)}")
            return None

# Mock service functions
def send_complaint_response(email_id: str, response: str):
    """Mock function to simulate sending a response to a complaint"""
    logger.info(f"Sending complaint response for email {email_id}")
    # In real implementation: integrate with email service


def send_standard_response(email_id: str, response: str):
    """Mock function to simulate sending a standard response"""
    logger.info(f"Sending standard response for email {email_id}")
    # In real implementation: integrate with email service


def create_urgent_ticket(email_id: str, category: str, context: str):
    """Mock function to simulate creating an urgent ticket"""
    logger.info(f"Creating urgent ticket for email {email_id}")
    # In real implementation: integrate with ticket system


def create_support_ticket(email_id: str, context: str):
    """Mock function to simulate creating a support ticket"""
    logger.info(f"Creating support ticket for email {email_id}")
    # In real implementation: integrate with ticket system


def log_customer_feedback(email_id: str, feedback: str):
    """Mock function to simulate logging customer feedback"""
    logger.info(f"Logging feedback for email {email_id}")
    # In real implementation: integrate with feedback system

def normalize_text(text):
    import re
    # Remove excessive whitespaces and normalize set ordering
    text = re.sub(r'\s+', ' ', text.strip())  # Remove line breaks and extra spaces
    set_matches = re.findall(r"\{[^{}]*\}", text)

    for match in set_matches:
        elements = sorted(e.strip() for e in match.strip('{}').split(','))
        sorted_set = '{' + ', '.join(elements) + '}'
        text = text.replace(match, sorted_set)

    return text

def save_file_version(filename, folder, filedata):
    """Function to save prompt file version"""
    try:
        # base version of the file
        version = 0

        # check the latest version of the file in the folder
        for file in os.listdir(folder):
            if filename in file:
                current_version = int(file.split(".")[0].split("_")[-1][1:])
                version = max(version, current_version)

        # compare the latest file to the new filedata provided - if no change exit out of the function
        if version != 0: # If there is a file in the folder - only then do the comparison. 
            with open(f"{folder}/{filename}_v{version}.txt", 'r') as f:
                data = normalize_text(f.read())

            if data == normalize_text(filedata):
                logger.info("There were no changes made to the prompts!")
                return None

        # Increment the version by 1 and create a new filename
        new_version  = str(version + 1)
        new_filename = f"{filename}_v{new_version}"

        # New file created with the updated prompt
        with open(f"{folder}/{new_filename}.txt", 'w', encoding='utf-8') as f:
            f.write(filedata)

        return None
    
    except Exception as e:
        # log error during upgrading the version of the file
        logger.info(f"Error upgrading the version of the file {filename}: {str(e)}")
        return None


def run_demonstration():
    """Run a demonstration of the complete system."""
    # Initialize the system
    processor = EmailProcessor()
    automation_system = EmailAutomationSystem(processor)

    # Process all sample emails
    results = []
    for email in sample_emails:
        logger.info(f"\nProcessing email {email['id']}...")
        result = automation_system.process_email(email)
        results.append(result)

    # Create a summary DataFrame
    df = pd.DataFrame(results)
    print("\nProcessing Summary:")
    print(df[["email_id", "success", "classification", "response_sent"]])
    # save results to a csv file
    df.to_csv("results.csv")

    return df


# Example usage:
if __name__ == "__main__":
    results_df = run_demonstration()
