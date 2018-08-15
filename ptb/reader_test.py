import ptb_reader as pr

source = "C:\\ptb\\ptb\\data"
train_data,valid_data,test_data,word_to_id,id_to_word = pr.ptb_raw_data(source)



for step,(x,y) in enumerate(pr.ptb_iterator(train_data,40,20)):
		print("y!!!!!!!!!!!!!!!!!!!!!!!")
		print(step)
		print(x)
		print(y)