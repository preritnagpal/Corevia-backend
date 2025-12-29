import ee
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Earth Engine
ee.Initialize(project=os.getenv("GEE_PROJECT_ID"))

print("GEE AUTH SUCCESS")
