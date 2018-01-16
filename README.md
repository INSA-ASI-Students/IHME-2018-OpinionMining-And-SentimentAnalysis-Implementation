# IHME-2018-OpinionMining-And-SentimentAnalysis-Implementation

## Lancement

```bash
$ cd src
$ virtualenv venv
$ source ./venv/bin/activate
$ pip3 install -r requirements.txt
$ python3 main.py --action="learn" --download-requirements
$ python3 main.py --action="predict"
$ perl ./eval_semeval16_task6_v2/eval.pl ./eval_semeval16_task6_v2/SemEval2016-Task6-subtaskA-testdata-gold.txt ./output.csv
```

## Options du script "main"

Il est possible de lancer le script avec plusieurs options:
* `--action=` pour déterminer l'action à executer, au choix: `predict`, `learn`
* `--fusion-dataset` afin de fusionner, mélanger, et redécouper les jeux de données lors d'un learn
* `--sentiment=` pour déterminer la méthode de sentiment analysis à utiliser, au choix: `learning`, `wordnetaffect`, `sentiwordnet`
* `--opinion=` afin de déterminer la méthode utilisée pour déterminer l'opinion, au choix: `neural_network`
* `--stance=` afin de déterminer la méthode utilisée pour déterminer le stance, au choix: `stance`
* `--learn-file=` afin de spécifier le path local du fichier d'apprentissage à utiliser
* `--test-file=` afin de spécifier le path local du fichier de test à utiliser
* `--predict-file=` afin de spécifier le path local du fichier de prediction à utiliser
* `--output-file=` afin de spécifier le path local du fichier ou seront écris les résultats à l'issue de la prediction
* `--download-requirements` afin de télécharger les dépendances
