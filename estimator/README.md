# Estimator CAG

API FastAPI que recibe una transcripción de reunión y devuelve una estimación de software generada por OpenAI. El servicio incluye ejemplos de estimaciones previas en el prompt como contexto CAG.

## Configurar y arrancar el proyecto

Necesitas `uv` y una clave de API de OpenAI. Desde el repositorio clonado, entra en `estimator` y copia `.env.example` si todavía no tienes un `.env`.

```cmd
cd estimator
copy .env.example .env
```

Edita `.env` y coloca tu clave en `OPENAI_API_KEY`. Estos son los valores usados en el ejercicio:

```dotenv
OPENAI_API_KEY=
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
APP_ENV=development
LOG_LEVEL=DEBUG
```

Arranca la aplicación desde `estimator`:

```cmd
uv run uvicorn app.main:app --reload
```

Abre [el endpoint de salud](http://127.0.0.1:8000/health) para comprobar que el servicio devuelve `status: healthy`.

## Probar el servicio con la transcripción del ejercicio

La [transcripción de ejemplo](transcripcion.json) contiene el texto de una reunión ficticia sobre una landing page con integración de HubSpot y un blog. El archivo se puede enviar directamente como cuerpo JSON a `POST /api/v1/estimate`.

Con el servidor en marcha, ejecuta este comando desde `estimator` en el Símbolo del sistema de Windows:

```cmd
curl.exe --json @transcripcion.json http://127.0.0.1:8000/api/v1/estimate
```

Desde el navegador, abre [Swagger UI](http://127.0.0.1:8000/docs), selecciona `POST /api/v1/estimate`, pulsa **Try it out**, pega el contenido completo del archivo y pulsa **Execute**. La respuesta JSON incluye `estimation`, `model`, `provider` y `usage`.

### Respuesta obtenida

```json
{
  "estimation": "## Landing Page con Formulario de Contacto e Integración con CRM\n\n### Task Breakdown\n\n| Task | Hours | Cost (EUR) |\n|------|------:|------------|\n| Revisión de requisitos y análisis de UX | 8 | 500 |\n| Adaptación del diseño de Figma a código | 16 | 800 |\n| Implementación del formulario de contacto | 10 | 625 |\n| Integración con el CRM de HubSpot | 14 | 875 |\n| Desarrollo de la sección de blog con editor WYSIWYG | 24 | 1,500 |\n| Configuración y pruebas de SEO básico | 8 | 500 |\n| Pruebas de funcionalidad y corrección de errores | 12 | 750 |\n| Documentación y despliegue final | 6 | 375 |\n\n### Totals\n\n- **Total hours:** 98\n- **Total cost:** 6,425 EUR\n\n### Recommended Team\n\n- 1 Desarrollador Frontend Senior\n- 1 Diseñador UI/UX (parte del tiempo durante la primera semana)\n\n### Estimated Duration\n\n**4 semanas**, asumiendo acceso a las credenciales de HubSpot e infraestructura necesaria para el despliegue.",
  "model": "gpt-4o-mini-2024-07-18",
  "provider": "openai",
  "usage": {
    "input_tokens": 1774,
    "output_tokens": 254,
    "total_tokens": 2028
  }
}
```

Las partidas suman 98 horas y 5.925 EUR. El modelo informó un coste total de 6.425 EUR.

## Estructura del proyecto

```
estimator/
├── app/
│   ├── main.py            # Aplicación FastAPI y endpoint de salud
│   ├── config.py          # Configuración con Pydantic Settings
│   ├── routers/
│   │   └── estimations.py  # Endpoint POST /api/v1/estimate
│   ├── services/
│   │   └── llm_service.py # Construcción del prompt y llamada al LLM
│   ├── schemas/
│   │   └── estimation.py   # Modelos Pydantic (request/response)
│   └── context/
│       └── examples.py    # Ejemplos de estimación para el contexto CAG
├── .env.example           # Variables de entorno del proyecto
├── transcripcion.json     # Entrada de ejemplo para POST /api/v1/estimate
└── pyproject.toml          # Dependencias del proyecto
```
