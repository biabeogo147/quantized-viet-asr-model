# 2026-05-20 AI Hub Package Cutover Results

## Scope

This refactor completed the clean AI Hub cutover after deployment packaging was already working:

- AI Hub workflow code now lives under `src/aihub/`
- public module names now describe responsibility instead of rollout phase names:
  - `aihub.session`
  - `aihub.evaluation`
  - `aihub.deployment`
- current workflow docs were renamed to `aihub-*`
- the notebook-layout test was renamed to `test_aihub_notebook_layout.py`
- no compatibility wrappers were left behind in `src/tools/`

Legacy identifiers were kept only where they are part of retained evidence or artifact continuity:

- notebook filename: `On_device_Ai_option1_pilots.ipynb`
- record groups such as `zipformer_encoder_option1` and `vpcd_option1_local_aimet`
- retained Zipformer compile input: `encoder.aihub.option1.onnx`

## Code And Docs Updated

- [session.py](/D:/DS-AI/BKMeeting-Research/python-model-test/src/aihub/session.py)
- [evaluation.py](/D:/DS-AI/BKMeeting-Research/python-model-test/src/aihub/evaluation.py)
- [deployment.py](/D:/DS-AI/BKMeeting-Research/python-model-test/src/aihub/deployment.py)
- [README.md](/D:/DS-AI/BKMeeting-Research/python-model-test/src/aihub/README.md)
- [On_device_Ai_option1_pilots.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/On_device_Ai_option1_pilots.ipynb)
- [overview.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/architecture/overview.md)
- [aihub-overview.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/workflows/aihub-overview.md)
- [aihub-rerun.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/workflows/aihub-rerun.md)
- [aihub-deployment.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/workflows/aihub-deployment.md)
- [android-handoff.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/workflows/android-handoff.md)
- [aihub-retained-lanes.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/qnn/aihub-retained-lanes.md)
- [test_aihub_session.py](/D:/DS-AI/BKMeeting-Research/python-model-test/test/test_aihub_session.py)
- [test_aihub_evaluation.py](/D:/DS-AI/BKMeeting-Research/python-model-test/test/test_aihub_evaluation.py)
- [test_aihub_deployment.py](/D:/DS-AI/BKMeeting-Research/python-model-test/test/test_aihub_deployment.py)
- [test_aihub_notebook_layout.py](/D:/DS-AI/BKMeeting-Research/python-model-test/test/test_aihub_notebook_layout.py)

## Verification

Environment used:

- `D:\Anaconda\envs\speech2text\python.exe`

Commands run successfully:

```powershell
& 'D:\Anaconda\envs\speech2text\python.exe' -m pytest test/test_aihub_session.py test/test_aihub_evaluation.py test/test_aihub_deployment.py test/test_aihub_notebook_layout.py -v -p no:cacheprovider
& 'D:\Anaconda\envs\speech2text\python.exe' -m compileall src
& 'D:\Anaconda\envs\speech2text\python.exe' -m aihub.deployment --project all --run-label 20260519-6pm --repo-root . --device-name "Samsung Galaxy S24 (Family)" --dry-run
& 'D:\Anaconda\envs\speech2text\python.exe' -m aihub.deployment --project all --run-label 20260519-6pm --repo-root . --device-name "Samsung Galaxy S24 (Family)"
```

Observed results:

- `55 passed` across the focused AI Hub and notebook-layout suites
- `python -m compileall src` passed
- deployment dry-run resolved the retained target ids:
  - `zipformer -> mqero78kn`
  - `vpcd -> mmxwpeyen`
- real deployment packaging succeeded for both retained projects

## Deployment Outputs

Current deployment packages now live at:

- [zipformer deployment package](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/deploy/zipformer/20260519-6pm)
- [vpcd deployment package](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/deploy/vpcd/20260519-6pm)

Key output files:

- [zipformer deployment_manifest.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/deploy/zipformer/20260519-6pm/deployment_manifest.json)
- [vpcd deployment_manifest.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/deploy/vpcd/20260519-6pm/deployment_manifest.json)
- [zipformer io_contract.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/deploy/zipformer/20260519-6pm/io_contract.json)
- [vpcd io_contract.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/deploy/vpcd/20260519-6pm/io_contract.json)

## Notes

- AI Hub still returned downloaded filenames with the doubled suffix `.onnx.onnx.zip`; those filenames were preserved as-downloaded.
- The real deployment run did not pass `--qairt-version`, so `deployment_manifest.json` currently records `qairt_version: null`.
- `On_device_Ai.ipynb` was already dirty in the worktree before this refactor and was left untouched.
