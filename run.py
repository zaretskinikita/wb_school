import src
import argparse
parser = argparse.ArgumentParser('''
	Запускает проект в зависимости от режима mode (Train / Inference / Default (оба режима последовательно))
	''')

requiredNamed = parser.add_argument_group('Required arguments')
requiredNamed.add_argument('-p', '--path', help='Path to the project', required=True)
parser.add_argument('-m', '--mode', help='Train (0) / Inference (1) mode / Default (2) (both train and inference)', default=2)
parser.add_argument('-d', '--data_name', help='Raw data file name', default='data.csv')
parser.add_argument('-l', '--label_column', help='Name of the label column', default='label')
parser.add_argument('-t', '--text_column', help='Name of the text column', default='text')
parser.add_argument('-md', '--model_name', help='Trained model name', default='LogRegr.pkl')


args = parser.parse_args()

src.make_processed_data(args.path, args.data_name, args.mode, args.label_column, args.text_column)
if args.mode == 0:
	src.train_model(args.path, args.data_name, args.label_column)
if args.mode == 1:
	src.predict_model(args.path, args.data_name, args.model_name)
if args.mode == 2:
	src.train_model(args.path, args.data_name, args.label_column)
	src.predict_model(args.path, args.data_name, args.model_name)