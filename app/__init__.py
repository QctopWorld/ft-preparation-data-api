from .api import app

if __name__ == "__main__":
    import uvicorn
    # Remplacez "mon_package" par le nom exact de votre module Python
    uvicorn.run("ft_preparation_donnees:app", host="127.0.0.1", port=9000, reload=True)