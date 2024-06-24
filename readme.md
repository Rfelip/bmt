Todos os arquivos necessários pra execução do programa estão contidos nessa pasta. O notebook, "run_everything.ipynb" contém uma execução ordenada do programa, 
e é equivalente a um main.py.

Achei o formato de notebook melhor por que ele permite ler os logs sem precisar rodar pelo terminal. Também permite alterar o modelo, e testar se ele consegue ler o modelo novo com todo o cuidado. O modelo *não cria* alguns dos caminhos pra output, então *tenha certeza de que a pasta result existe* quando você executar o notebook.

Os logs também são salvos na pasta log.log.

Stemming está disponível nas configurações, e deve ser adicionado no GLI.CFG, o config do índice. *não misture listas inversas e modelos vetoriais sem stemming com os com stemming*

Infelizmente, a parte de avaliação não ficou pronta a tempo. Mas o stemming e o carregamento dos dados de resultado está disponível em "see_every_graph.ipynb". Uma função, inicial, de precisão e recall da informação também está lá. 