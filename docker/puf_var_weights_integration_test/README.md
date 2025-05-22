# Integration test for a full run of the data pipeline from scratch
```
PIPELINE_NAME=test-pipeline-5-21-b
sudo docker build -t $PIPELINE_NAME .
``

To run the script in the entrypoint:
```
sudo docker run --rm -t -e HUGGING_FACE_TOKEN=$HUGGING_FACE_TOKEN $PIPELINE_NAME
```

Or. to run interactively,
```
sudo docker run --rm -it -e HUGGING_FACE_TOKEN=$HUGGING_FACE_TOKEN $PIPELINE_NAME /bin/bash
```
