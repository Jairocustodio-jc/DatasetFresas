# Etiquetado del dataset de 5 estados

Copia versionada del trabajo de etiquetado que vive en
`D:\新草莓\DatasetId_360753_1652783343`. Las imágenes a resolución completa no
están aquí porque pesan 9,8 GB en PNG y otros 4,9 GB en el export YOLO, muy por
encima de lo que admite GitHub. Sí están las anotaciones, que son lo que costó
trabajo, y el pipeline que las produjo.

## Qué hay

- `anotaciones/` — 3 099 JSON de AnyLabeling con las 16 089 cajas, su etiqueta,
  los atributos por caja y las banderas. Es el entregable.
- `yolo/` — las 3 099 etiquetas en formato YOLO más el `data.yaml`. Sin imágenes.
- `_work/` — los 29 scripts del pipeline de medición y corrección.
- `informe/` — INFORME.md, los CSV de cambios y atributos, y las hojas de
  contacto de la auditoría de la frontera del 75 %.

## Clases

Índices YOLO 0 a 4, en español desde el 3-11-2026:

| 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| verde_inmadura | rosa_temprano | comercial_basico | comercial_alto | madurez_plena |

El corte entre `comercial_basico` y `comercial_alto` está en el 75 % de
superficie visible coloreada, siguiendo USDA No. 1, y no en el 80 % de
NY/T 2787-2015. Fue una decisión consciente.

## Advertencias

- No ejecutar `_work/apply.py`. Reconstruye desde `data_anylabeling_2` y borraría
  las 184 correcciones que vinieron de la revisión en el celular.
- La medición usa GrabCut, que toma el RNG global de OpenCV. Sin `cv2.setRNGSeed`
  fijo el resultado cambiaba en torno al 6 % de las cajas entre ejecuciones.
- Los lotes de `revision/`, que son con los que entrena Kaggle, se congelaron el
  3-10-2026 y por tanto no llevan esas 184 correcciones ni los nombres en
  español. Las etiquetas de aquí sí.
