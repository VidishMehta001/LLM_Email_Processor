import os
from openai import OpenAI
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticEmail(object):

    # Initialize the class object
    def __init__(self):
        self.valid_categories = [
            "complaint", 'inquiry', 'feedback', 'support_request', 'other'
        ]
        self.spectrum_info = [
            "Please ensure that the email samples generated range from reporting issue and mild disappointment to extreme frustration and agression",
            "Please ensure that the email samples are simple FAQ questions as well as more complex multi-step inquiries",
            "Please ensure that the email samples are from constructive criticism to a lot of praise and happiness for the product",
            "Please ensure that the support request are simple request to more complex multi-step technical requests",
            "Please ensure that the other are from internal teams to external collaborators and partners"
        ]
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Main generate samples function
    def generate_samples(self, number_of_samples=1):
        """
        Generates synthetic email samples using Open AI API

        Args: 
            number of samples(int): number of samples to be created
        
        Returns:
            list of generated samples
        """
        logger.info(f"Generating {number_of_samples} number of samples")
        generated_samples = []

        # loop on the valid category
        for i, category in enumerate(self.valid_categories):
            logger.info(f"Evaluating synthetic samples for {self.valid_categories[i]} category")
            for j in range(number_of_samples):
                logger.info(f"Generating synthetic samples for {str(j+1)} sample")

                # setup the prompt required for generating the email smaples. 
                if self.valid_categories[i] != "other":
                    prompt = f"""
                    Generate one realistic email from customer in json object format with the following information as keys and values generated:
                    id: 
                    from:
                    subject:
                    body:
                    timestamp:
                    category: {self.valid_categories[i]}
                    The email should be of category {self.valid_categories[i]}
                    {self.spectrum_info[i]}
                    Please return email as a json object
                    """
                else:
                    prompt = f"""
                    Generate one realistic email from internal teams or external collaborators in json object format with the following information as keys and values generated:
                    id: 
                    from:
                    subject:
                    body:
                    timestamp:
                    category: {self.valid_categories[i]}
                    The email should be of category {self.valid_categories[i]}
                    {self.spectrum_info[i]}
                    Please return email as a json object
                    """
                
                # Generate synthetic samples by calling the open AI api
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Imagine you are an expert in creating synthetic emails"},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type":"json_object"}
                )

                # Parse the response
                response = response.choices[0].message.content
                
                # Load the json object
                response_json = json.loads(response)

                # clean the response body - replace line breaks
                response_text = response_json['body'].replace("\n\n", " ")
                response_text = response_text.replace("\n", " ")

                # clean response body
                response_text = " ".join(response_text.split())

                # Build the sample email template
                sample = {
                    "id": f"gen-{category}-{j+1}",
                    "from": response_json.get("from", f"customer_{j+1}@example.com"),
                    "body": response_text,
                    "timestamp": response_json.get("timestamp"),
                    "category": response_json.get("category", category)
                }

                # Append the generated samples with the response_json
                logger.info(f"Saving sample email for {self.valid_categories[i]} and {str(j+1)} sample")
                generated_samples.append(sample)

        return generated_samples


if __name__ == "__main__":
    # Run the generated samples function
    generated_samples = SyntheticEmail().generate_samples(number_of_samples=4)
    # save the samples
    with open('generated_samples.json', 'w') as f:
        json.dump(generated_samples, f, indent=4)


