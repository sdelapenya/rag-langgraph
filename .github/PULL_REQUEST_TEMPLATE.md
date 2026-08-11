<!--
  Plantilla de PR. Borra las secciones que no apliquen: un PR corto con dos
  secciones vacías se lee peor que uno con solo las que tienen algo que decir.
  Estos comentarios no se ven en el PR publicado.
-->

## Qué entra

<!-- Una frase de contexto y luego la tabla: qué cambia y dónde mirarlo. -->

| | Dónde |
|---|---|
| | |

## Decisiones

<!--
  La sección que más pesa. Cada punto: la decisión, la alternativa que
  descartaste y por qué. Si un cambio no tuvo alternativa, no va aquí.
-->

-

## Comprobado

<!-- Comandos reales con su resultado, no "lo he probado". -->

```
ruff check .              →
python -m pytest          →
docker compose up         →
```

## Riesgo

<!--
  Qué se rompe si esto está mal y cómo se vuelve atrás. En cambios que tocan
  la recuperación o el prompt, incluir las métricas antes/después medidas con
  evaluate.py — no la impresión de que "responde mejor".
-->
