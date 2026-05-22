##  Requisitos previos

Antes de ejecutar el proyecto, asegúrate de tener instalado:

- Python 3.10 o superior  

## Descarga del proyecto

Clona este repositorio:

```bash
git clone https://github.com/tu-usuario/tu-repo.git
```

O descárgalo como archivo `.zip` y descomprímelo.


##  Configuración del modelo de IA

El modelo de IA se puede descargar desde el siguiente enlace:

```bash
https://huggingface.co/mozilla-ai/Mistral-7B-Instruct-v0.2-llamafile/blob/main/mistral-7b-instruct-v0.2.Q2_K.llamafile
```
Para correrlo en local se debe entrar a una consola y abrir la carpeta en la que se encuentra dicho archivo
y ejecutar el siugiente comando:
```bash
.\mistral-7b-instruct-v0.2.Q2_K.exe --server --host 0.0.0.0 --port 8080
```

### Activar el entorno virtual

#### Windows (PowerShell):

```bash
venv\Scripts\Activate.ps1
```

#### Windows (CMD):

```bash
venv\Scripts\activate
```

#### Linux / Mac:

```bash
source venv/bin/activate
```

---

## Ejecución del proyecto

Con el modelo de IA ejecutandose en local ejecuta el programa principal:

```bash
modelo_interfaz.py
```

---
