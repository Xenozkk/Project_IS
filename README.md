# Streamlit Wine Dataset Project

## Project Setup

### Prerequisites

- Python 3.10.11 or higher
- pip
- virtualenv

 ### Installation Steps

1. Clone the repository

   You can clone the repository by running the following command in your terminal:
   git clone <repository-url>


2. Create a virtual environment

    python -m venv venv

    py -3.10 -m venv venv

3. Activate virtual environment

    # On Windows
    venv\Scripts\activate

    # On Linux or macOS
    source venv/bin/activate

4. Install dependencies

    # If you have a requirements.txt file:
    pip install -r requirements.txt

    # Alternatively, if you have a setup.py file:
    pip install -e .

5. Run the Streamlit app

    streamlit run src/main.py
