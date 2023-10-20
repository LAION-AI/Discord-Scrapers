---
dataset_info:
  features:
  - name: caption
    dtype: string
  - name: image
    dtype: image
  - name: link
    dtype: string
  - name: message_id
    dtype: string
  - name: timestamp
    dtype: string
  splits:
  - name: train
    num_bytes: 0
    num_examples: 0
  download_size: 0
  dataset_size: 0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

Use the Edit dataset card button to edit.