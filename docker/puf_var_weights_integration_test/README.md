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


Detach: Ctrl+P, Ctrl+Q
Reattach: sudo docker attach wizardly_joliot
New shell: sudo docker exec -it wizardly_joliot /bin/bash

# restart  get back in
sudo docker restart wizardly_joliot
sudo docker exec -it wizardly_joliot /bin/bash

# Stop and start fresh (you'll lose any unsaved work in the container)
sudo docker stop wizardly_joliot
sudo docker start -i wizardly_joliot
