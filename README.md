# Offline AI-Powered Document OCR Platform 📄🧠

## Overview

This project is an **offline-capable Optical Character Recognition (OCR) and data extraction platform** designed for processing various identity and generic documents. It runs entirely on your local machine, ensuring **data privacy** while delivering high accuracy through AI-powered pipelines.

## ✨ Features

* **Offline Operation**: All AI models run locally using [Ollama](https://ollama.com), no internet required after setup.
* **Multi-Document Support**: Handles Passports, Driving Licenses, Aadhaar Cards, Emirates IDs, and generic documents.
* **Universal File Input**: Supports JPG, PNG, PDF.
* **High-Accuracy OCR**:

  * Image pre-processing for optimal OCR.
  * PaddleOCR for initial text extraction.
  * MiniCPM-V 2.6 (Ollama) for intelligent correction.
* **Smart Data Extraction**: Uses dynamic JSON templates with `additional_data` for unique info.
* **Face & Image Extraction**: Crops and saves detected faces and original images.
* **Asynchronous Processing**: Celery workers for background OCR tasks.
* **Database Storage**: PostgreSQL stores structured data and images.
* **User-Friendly Web UI**: HTML/CSS-based document submission & results.
* **Document History**: Browse and revisit processed documents.
* **RESTful API**: `/api/v1/extract` for integration.
* **Swagger API Docs**: `/api/docs` for interactive API testing.
* **Load Balancing**: Nginx reverse proxy distributes requests.
* **Dockerized Deployment**: Easy setup via Docker Compose.

## 📦 Architecture

```
+----------------+       +-------------------+       +--------------------+
|                |       |  Flask App (Nginx)|       |  Celery Worker(s)  |
|  User / Other  |------>|      (Gunicorn)   |<----->|  (Async Tasks)     |
|   Application  |       |                   |       | + PaddleOCR + AI   |
+----------------+       +-------------------+       +--------------------+
        ^                        ^                          ^
        | API Requests           | Task Queue               |
        |                        |                          |
        |                        v                          |
        |                     +----------+                  |
        |                     |PostgreSQL|<-----------------+
        |                     +----------+
        |                          ^
        +--------------------------+
                   |
                   v
               +--------+
               |Ollama  |
               +--------+
```

## ⚙️ Prerequisites

* [Docker](https://docs.docker.com/get-docker/)
* Docker Compose (bundled with Docker Desktop)

## 🚀 Setup & Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/id_scanner_app.git
   cd id_scanner_app
   ```

2. **Download Haar Cascade File**

   ```bash
   wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml -P ocr_app/
   ```

3. **Create `.env` File**

   ```env
   POSTGRES_DB=ocr_db
   POSTGRES_USER=ocr_user
   POSTGRES_PASSWORD=your_strong_password
   ```

4. **Make Ollama Entrypoint Executable**

   ```bash
   chmod +x ollama/entrypoint.sh
   ```

5. **Build & Start Services**

   ```bash
   docker compose up --build -d
   ```

6. **Monitor Logs**

   ```bash
   docker compose logs -f ocr_ollama
   docker compose logs -f ocr_celery_worker
   docker compose logs -f ocr_flask_app
   ```

## 🚀 Usage

### Web UI

* Access: [http://localhost](http://localhost)
* Upload documents, choose type, view results, and check history.

### API

* Docs: [http://localhost/api/docs](http://localhost/api/docs)
* Example:

  ```bash
  curl -X POST \
    http://localhost/api/v1/extract \
    -F 'doc_type=Passport' \
    -F 'files=@/path/to/passport.pdf'
  ```

## ❓ Troubleshooting

* **Ollama model pull error**:

  ```bash
  docker compose down -v
  ollama pull minicpm-v:8b
  docker compose up --build -d
  ```
* **OOM errors**: Increase Docker memory allocation (8GB+).
* **Swagger not loading**: Ensure `ocr_app/static/swagger.json` exists.

## ⏩ Future Enhancements

* GPU acceleration.
* More document templates.
* Error reporting integration.
* User authentication.
* Improved UI/UX.

---
