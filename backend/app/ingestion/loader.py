import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from io import BytesIO
from fastapi import UploadFile

class DocumentLoader:
    @staticmethod
    async def load_pdf(file: UploadFile) -> str:
        content = await file.read()
        pdf_file = BytesIO(content)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    @staticmethod
    async def load_txt(file: UploadFile) -> str:
        content = await file.read()
        return content.decode("utf-8")

    @staticmethod
    def load_url(url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Remove scripts and styles
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
                
            text = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
        except Exception as e:
            raise ValueError(f"Failed to load URL: {str(e)}")
