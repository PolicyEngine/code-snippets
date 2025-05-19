# Integration test for a full run of the data pipeline from scratch

```
sudo docker build -t test-pipeline-1 .
sudo docker run --rm -t -e HUGGING_FACE_TOKEN=$HUGGING_FACE_TOKEN test-pipeline-1
```
