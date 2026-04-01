# Python-Model-Test Plans

Thu muc nay luu cac implementation plan o cap repo cho `python-model-test`.

Nguyen tac:
- Uu tien plan theo phase ro rang va co ngay thang trong ten file.
- Neu mot plan la phase dau cho Android, ghi ro artifact dau ra de repo `bkmeeting` co the consume ve sau.
- Khi mot plan phu thuoc vao plan truoc, ghi ro dependency thay vi tron chung vao mot file qua lon.

Plan hien tai:
- `2026-03-26-quantize-vpcd.md`: ke hoach cu cho quantize punctuation model theo huong Snapdragon 8 Gen 2.
- `2026-04-01-zipformer-python-first-bundle-qnn.md`: ke hoach canonical hien tai. File nay da duoc mo rong thanh architecture plan cho shared `model_bundle` + multi-project `quantize`, dung cho ca `vpcd` va `zipformer`, va bao gom phase Zipformer `fixed-shape + PTQ + QDQ` candidate bundle.
