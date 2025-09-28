import os
from dotenv import load_dotenv
from fyers_apiv3.fyersModel import SessionModel

load_dotenv()

def generate_access_token():
    client_id = os.getenv("CLIENT_ID")
    secret_key = os.getenv("SECRET_KEY")
    redirect_uri = os.getenv("REDIRECT_URI")

    session = SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code"
    )

    auth_code_url = session.generate_authcode()

    print("Please open the following URL in your browser, log in, and then paste the auth_code from the redirect URL:")
    print(auth_code_url)

    auth_code = input("Enter the auth_code: ")

    session.set_token(auth_code)
    access_token_response = session.generate_token()

    if access_token_response.get("access_token"):
        # Update the .env file
        with open(".env", "r") as f:
            lines = f.readlines()
        with open(".env", "w") as f:
            for line in lines:
                if line.startswith("ACCESS_TOKEN="):
                    f.write(f"ACCESS_TOKEN={access_token_response['access_token']}\n")
                else:
                    f.write(line)
        print("Access token generated and updated successfully.")
    else:
        print("Failed to generate access token.")
        print("Response:", access_token_response)

if __name__ == "__main__":
    generate_access_token()
