#!/bin/bash
# Vision-Restore AI - API Client Test Script
# Tests the API endpoints using curl

API_URL="${API_URL:-http://localhost:8000}"

echo "========================================"
echo "Vision-Restore AI - API Test Script"
echo "========================================"
echo ""
echo "API URL: $API_URL"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Health Check
echo -e "${YELLOW}Test 1: Health Check${NC}"
echo "GET $API_URL/health"
curl -s "$API_URL/health" | jq '.'
echo ""

# Test 2: Service Status
echo -e "${YELLOW}Test 2: Service Status${NC}"
echo "GET $API_URL/api/v1/status"
curl -s "$API_URL/api/v1/status" | jq '.'
echo ""

# Test 3: Get Available Models
echo -e "${YELLOW}Test 3: Get Available Models${NC}"
echo "GET $API_URL/api/v1/models"
curl -s "$API_URL/api/v1/models" | jq '.'
echo ""

# Test 4: Enhance Image (if test image exists)
if [ -f "test_image.jpg" ]; then
    echo -e "${YELLOW}Test 4: Enhance Image${NC}"
    echo "POST $API_URL/api/v1/enhance"
    
    curl -X POST "$API_URL/api/v1/enhance" \
        -F "file=@test_image.jpg" \
        -F "scale=2" \
        -F "upscale_model=realesrgan" \
        -F "face_enhance=true" \
        -F "return_base64=false" \
        -o "test_output.png" \
        -w "\nStatus: %{http_code}\nTime: %{time_total}s\n"
    
    if [ -f "test_output.png" ]; then
        echo -e "${GREEN}✓ Output saved to test_output.png${NC}"
        file test_output.png
    else
        echo -e "${RED}✗ Failed to save output${NC}"
    fi
    echo ""
else
    echo -e "${YELLOW}Test 4: Skipped (no test_image.jpg found)${NC}"
    echo ""
fi

# Test 5: Enhance with Base64 Response
if [ -f "test_image.jpg" ]; then
    echo -e "${YELLOW}Test 5: Enhance with Base64 Response${NC}"
    echo "POST $API_URL/api/v1/enhance (with return_base64=true)"
    
    response=$(curl -s -X POST "$API_URL/api/v1/enhance" \
        -F "file=@test_image.jpg" \
        -F "scale=2" \
        -F "upscale_model=realesrgan" \
        -F "face_enhance=false" \
        -F "return_base64=true")
    
    echo "$response" | jq 'del(.data) | .'
    echo ""
else
    echo -e "${YELLOW}Test 5: Skipped (no test_image.jpg found)${NC}"
    echo ""
fi

echo "========================================"
echo "Tests Complete"
echo "========================================"
