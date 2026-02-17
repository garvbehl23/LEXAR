# NYAYAVIEW: A Streamlit Application for Legal Question Answering

NYAYAVIEW is a research conference demo frontend that integrates with the LEXAR backend model to provide an interactive platform for legal question answering. This application leverages advanced natural language processing techniques to retrieve and generate legal information based on user queries.

## Features

- **User-Friendly Interface**: A clean and intuitive layout for easy navigation.
- **Legal Query Input**: Users can input legal questions and receive detailed answers.
- **Evidence Display**: Retrieved statutory evidence is presented in a structured format.
- **Token-Level Provenance**: Users can view the source of each generated token, ensuring transparency in the answers provided.
- **Confidence Scores**: The application displays confidence levels for the generated answers, helping users assess the reliability of the information.

## Installation

To set up the NYAYAVIEW application, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nyayaview.git
   cd nyayaview
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run src/app.py
   ```

## Usage

Once the application is running, you can:

- Enter a legal question in the input box.
- Click the "Analyze" button to retrieve the answer.
- View the generated answer along with the relevant statutory evidence and token provenance.

## Project Structure

The project is organized as follows:

```
nyayaview
├── src
│   ├── app.py                     # Main entry point for the Streamlit app
│   ├── components                  # UI components for the application
│   ├── services                    # Services for interacting with the LEXAR backend
│   ├── utils                       # Utility functions for formatting and validation
│   └── config                      # Configuration settings for the app
├── assets                          # Static assets like styles and images
├── requirements.txt                # Python dependencies
├── .streamlit                      # Streamlit configuration
└── README.md                       # Project documentation
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

**Last Updated**: February 2026  
**Version**: 1.0.0