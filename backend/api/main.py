import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File

from backend.api.mock_data import MOCK_RESULTS
from backend.perception.config import PerceptionConfig
from backend.perception.pipeline import PerceptionPipeline


app = FastAPI(
    title="Temporal Vision API",
    version="1.0.0",
)


OUTPUT_DIR = Path("output")
INPUT_DIR = Path("input")

INPUT_DIR.mkdir(exist_ok=True)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "temporal-vision",
    }


def load_json_file(path: Path):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/results")
def get_results(mock: bool = False):
    if mock:
        return MOCK_RESULTS

    return {
        "perception": load_json_file(
            OUTPUT_DIR / "perception_output.json"
        ),
        "events": load_json_file(
            OUTPUT_DIR / "events.json"
        ),
        "explanation": load_json_file(
            OUTPUT_DIR / "explanation.json"
        ),
    }


@app.post("/run")
async def run_pipeline(
    video: UploadFile = File(...)
):
    video_path = INPUT_DIR / video.filename

    with video_path.open("wb") as file:
        file.write(await video.read())

    config = PerceptionConfig(
        input_video=str(video_path),
        output_dir=str(OUTPUT_DIR),
    )

    pipeline = PerceptionPipeline(config)
    pipeline.run()

    return {
        "status": "completed",
        "video": video.filename,
        "message": "Perception pipeline completed successfully.",
    }