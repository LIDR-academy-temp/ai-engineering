# Conclusiones — Sesión 10: búsqueda híbrida y reranking

> **Pendiente de revisión y firma.**
> Las etiquetas de relevancia del golden set las propuso Claude (antes de ver ningún
> resultado de retrieval) y están **pendientes de revisión humana**. Si cambian, hay que volver
> a ejecutar la evaluación y revisar estas conclusiones.

Fuentes: `results/retrieval_eval_20261109T134428Z.md` (golden set endurecido, `golden_set.json`)
y `results/retrieval_eval_20261109T133158Z.md` (primer golden set, `golden_set_v1_easy.json`).

## Resultados

| Config | Search | Reranking | Precision@5 (mean) | Distinct relevant docs in top-5 (mean) | Latency p50 (ms) | Latency p95 (ms) |
|---|---|---|---|---|---|---|
| A | vector | no | 0.92 | 2.00 | 5.4 | 6.1 |
| B | hybrid | no | 0.92 | 2.00 | 7.3 | 8.2 |
| C | vector | yes | 0.68 | 2.00 | 6456.7 | 7975.8 |
| D | hybrid | yes | 0.68 | 2.00 | 6511.9 | 7443.6 |

Precision@5 por consulta (techo = 1,00 en todas):

| Query | A | B | C | D |
|---|---|---|---|---|
| h1 sanidad (vídeo + wearables) | 1.00 | 1.00 | 1.00 | 1.00 |
| h2 fintech (onboarding) | 1.00 | 0.80 | 0.40 | 0.40 |
| h3 moda (webshop + devoluciones) | 1.00 | 1.00 | 1.00 | 1.00 |
| h4 COBOL / strangler (términos exactos) | 0.60 | 1.00 | 0.60 | 0.60 |
| h5 logística (robots + mantenimiento) | 1.00 | 0.80 | 0.40 | 0.40 |

El primer golden set (una consulta = un presupuesto dominante) **saturó la métrica**: las cuatro
configuraciones obtuvieron 0,68 de precision@5, exactamente el techo alcanzable. Por eso se
sustituyó por consultas parafraseadas con dos presupuestos relevantes y trampas de vocabulario
compartido. Ese primer resultado se conserva solo como referencia.

## 1. ¿Qué configuración usaríamos y por qué?

**B: búsqueda híbrida sin reranking.**

- En media empata con la vectorial (0,92 frente a 0,92), pero **gana donde la rama léxica tiene
  sentido**: en la consulta con términos técnicos exactos (h4: COBOL, strangler, anti-corruption
  layer, OAuth) sube de 0,60 a 1,00, al recuperar el segundo presupuesto relevante que la
  vectorial dejaba fuera.
- A cambio pierde 0,20 en h2 y h5, donde la rama léxica cuela un distractor en la posición 3 o 5.
  En ningún caso pierde un presupuesto relevante: los documentos relevantes distintos en el top-5
  se mantienen en 2,00.
- Su coste es despreciable: +1,9 ms de p50 (7,3 frente a 5,4 ms).
- Las peticiones de estimación reales contienen siglas y nombres de tecnologías (PSD2, HL7, FHIR,
  KYC…), justo el caso en que la híbrida aporta. Con un empate en media y una ganancia clara en
  ese tipo de consulta, la híbrida es la opción por defecto más razonable.

## 2. ¿Compensa la latencia del reranking en este caso de uso?

**No, con este modelo y este corpus.** Y la razón principal no es la latencia, sino la relevancia:

- **El reranking empeora la precisión**: de 0,92 a 0,68 de media. En h2 y h5 cae de 1,00 a 0,40
  porque coloca arriba las trampas: el KYC de vendedores del marketplace y el onboarding del
  robo-advisor en h2; la red eléctrica, el core bancario y el robo-advisor en h5. El cross-encoder
  (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, entrenado con pasajes cortos de MS MARCO) premia
  la coincidencia de una frase concreta de la consulta con un componente, no el encaje del proyecto
  completo.
- **La latencia, por sí sola, sería asumible**: unos 6,5 s de p50 (6,5–8,0 s de p95) para 50
  candidatos en CPU, frente a una generación con `gpt-5` y razonamiento alto que tarda del orden de
  uno a dos minutos. Supondría aproximadamente un 5–10 % del tiempo total de la petición. Pero no
  tiene sentido pagar ni un segundo por un paso que reduce la relevancia.
- **C y D salen idénticos en todas las consultas.** Con 60 chunks en el corpus base y un pool de
  50 candidatos, el reranker ordena prácticamente el corpus entero, así que la forma de hacer el
  recall deja de importar. En este corpus no podemos medir el valor real de recall-then-rerank.

## Limitaciones

- **Solo 5 consultas** (más 5 del primer golden set, que saturó la métrica). Una consulta mueve la
  media 0,04–0,12 puntos: las diferencias son orientativas, no estadísticamente significativas.
- **Anotación de una sola persona**, que además es la IA que diseñó las consultas. Está pendiente
  de revisión humana.
- **Corpus diminuto**: 17 presupuestos y 60 chunks. Un pool de 50 equivale casi a reordenar todo.
- **Precision@5 a nivel de chunk**: varios chunks del mismo presupuesto cuentan por separado. Por
  eso se reporta también el número de presupuestos relevantes distintos (2,00 en todas las
  configuraciones) y un techo por consulta.
- **Corpus y consultas en inglés**, con configuración full-text `english`, en lugar de `spanish`
  como pedía el enunciado: el corpus entregado está en inglés.
- **La rama léxica usa OR entre términos** (ordenado por `ts_rank_cd`). Con AND, una descripción
  completa de un proyecto no coincidía con ningún chunk y la híbrida era idéntica a la vectorial.
- **Latencia medida en local**: portátil con WSL2, 14 CPU lógicas, sin GPU, en un contenedor de un
  solo uso sin el proceso de recarga de uvicorn, que competía por la CPU. Medido de forma aislada,
  el modelo tarda 1,3–2,9 s para esos 50 pares, así que la cifra depende mucho del entorno.
- **Solo se mide retrieval**: no evaluamos si la estimación final mejora o empeora.

## Trabajo futuro (fuera del alcance de esta entrega)

- Revisar las etiquetas y ampliar el golden set (más de 20 consultas, varias personas anotando).
- Probar el reranking sobre un corpus mayor (con el corpus de tareas, 282 chunks) o con un pool
  menor que el corpus, para medir de verdad recall-then-rerank.
- Probar un cross-encoder más fuerte o puntuar con el resumen del proyecto completo en lugar del
  componente aislado.
- Medir precision a nivel de presupuesto deduplicado, además de a nivel de chunk.
