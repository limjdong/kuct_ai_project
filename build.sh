#!/bin/bash

# Render 배포 시 실행되는 빌드 스크립트
echo "🚀 Starting build process..."

# Python 패키지 설치
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# PDF 임베딩 생성
echo "📖 Creating embeddings from PDF..."
python create_embeddings.py

echo "✅ Build completed successfully!"
