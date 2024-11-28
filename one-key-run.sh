1. 运行frontmatter 进行静态分析获得将结果保存到data目录中，命名为xxx.json
2. 运行gen_setg.py 进行semantic genration，结果保存到data目录中，命名为setg_xxx.csv
3. 运行GAT_classifier目录下的test.py进行判断，结果保存到data目录中，结果会显示在terminal


java -jar ./frontmatter/frontmatter-1.0.8-SNAPSHOT.jar all -a /opt/Android/Sdk/platforms  --boomerang-timeout 3 -t  -i ./data/1b76d42256ec8e9ce6b699aeef85fbbb.apk -o ./data/1b76d42256ec8e9ce6b699aeef85fbbb.json
python gen_setg.py
python ./GAT_classifier/test.py