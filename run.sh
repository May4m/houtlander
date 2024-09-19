# export DEV_MODE=1
# export DATA_PATH
streamlit run app.py --server.port 8000 --server.address 0.0.0.0 --runner.fastReruns True --runner.postScriptGC True --global.developmentMode False
