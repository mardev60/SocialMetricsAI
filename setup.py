from setuptools import setup, find_packages

setup(
    name="socialmetricsai",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "flask",
        "mysql-connector-python",
        "scikit-learn",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "python-crontab",
    ],
    author="Marouan Makil",
    author_email="makil.uspn@gmail.com",
    description="API d'Analyse de Sentiments pour les tweets",
    keywords="sentiment analysis, nlp, tweets",
    python_requires=">=3.8",
) 