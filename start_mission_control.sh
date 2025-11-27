#!/bin/bash
echo "Starting Avalanche Mission Control..."
streamlit run src/ui/app.py --server.port 8504
read -p "Press enter to exit"
