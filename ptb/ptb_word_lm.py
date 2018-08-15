from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import ms.version
# ms.version.addpkg('six',"1.9.0")
# ms.version.addpkg("numpy","1.9.2")
# ms.version.addpgk("tensorflow","0.6.0")

import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
import pdb
import tensorflow.python.platform

import numpy as np
import tensorflow as tf


# from tensorflow.models.rnn import rnn_cell
# from tensorflow.models.rnn import seq2seq
import ptb_reader
import ptb_server


def data_type():
	return tf.float16 if FLAGS.use_fp16 else tf.float32


flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
	"model","small",
	"A type of model.Possible options are:small, medium, large.")
flags.DEFINE_string("data_path",None,"data_path")
flags.DEFINE_string("interactive",None,"interavtive")
flags.DEFINE_bool("use_fp16",False,"Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS
class PTBModel(object):
	def __init__(self,is_training,config,is_query=False,is_generative=False):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		size = config.hidden_size
		vocab_size = config.vocab_size
		#one input word is projected into hidden_size space using embeddings

		self._input_data = tf.placeholder(tf.int32,[batch_size,num_steps])
		self._prior_output =tf.placeholder(tf.int32,[batch_size,size])
		self._targets = tf.placeholder(tf.int32,[batch_size,num_steps])

		#Slightly better results can be obtained with forget gate biases
		#initialized to 1 but the hyperparameters of the model would need to be
		#different than reported in the paper

		lstm_cell = tf.contrib.rnn.BasicLSTMCell(size,forget_bias =0.0,state_is_tuple=False)
		if is_training and config.keep_prob < 1:
			lstm_cell = tf.contrib.rnn.DropoutWrapper(
				lstm_cell,output_keep_prob=config.keep_prob)
		cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*config.num_layers,state_is_tuple=False)

		self._initial_state = cell.zero_state(batch_size,tf.float32)
		print("model:_initial_state:")
		print(self._initial_state)

		#if not is_generative:
		with tf.device("/cpu:0"):
			embedding = tf.get_variable(
				"embedding", [vocab_size, size], dtype=data_type())
			inputs = tf.nn.embedding_lookup(embedding, self._input_data)



		if is_training and not is_query and is_generative and config.keep_prob < 1:
			inputs = tf.nn.dropout(inputs,config.keep_prob)




		outputs =[]
		print("embedding")
		print(embedding)
		states = []
		state = self._initial_state
		with tf.variable_scope("RNN"):
			for time_step in range(num_steps):
				if time_step > 0: tf.get_variable_scope().reuse_variables()
				(cell_output,state) = cell(inputs[:,time_step,:],state)
				outputs.append(cell_output)
				states.append(state)	
		#output dimension is batch_sizeXhidden_size
		outputs = tf.concat(outputs,1)
		output = tf.reshape(outputs,[-1,size])
		#output tf.reshape(tf.concat(1,outputs),[-1,size])
		#logit dimension is batch_sizeX
		logits =tf.nn.xw_plus_b(
							output,
							tf.get_variable("softmax_w",[size,vocab_size]),
							tf.get_variable("softmax_b",[vocab_size]))
		self._logits = logits
		self._outputs = outputs
		self._output = output
		self._inputs = inputs
		self._final_state = states[-1]
		logits = tf.reshape(logits,[self.batch_size,self.num_steps,vocab_size])
		#tf.reshape(self._targets,[-1]),
		transfer_target = tf.reshape(self._targets,[self.batch_size,self.num_steps])
		#self._targets = tf.reshape(self._targets,[self.batch_size,self.num_steps])
		weights = tf.ones([self.batch_size,self.num_steps],dtype = data_type())
		if is_query or is_generative:
			#slef._loigts = tf.matmul(output,tf.get_variable("softmax_w"))+tf.get_variable("softmax_b")
			probs = tf.nn.softmax(logits)
			self._probs = probs
			top_k = tf.nn.top_k(probs,20)[1]
			self._top_k = top_k
			return
		else:
			loss = tf.contrib.seq2seq.sequence_loss(logits,
												transfer_target,
												weights)
			self._cost = cost = tf.reduce_sum(loss)/batch_size
			if not is_training:
				return
		self._lr = tf.Variable(0.0,trainable = False)
		tvars = tf.trainable_variables()
		grads,_ = tf.clip_by_global_norm(tf.gradients(cost,tvars),
			config.max_grad_norm)
		optimizer = tf.train.GradientDescentOptimizer(self.lr)
		self._train_op = optimizer.apply_gradients(zip(grads,tvars))


		print("shape of initial_state:")
		print(self._initial_state)


		print("the end!!!!!")

	def assign_lr(self,session,lr_value):
		session.run(tf.assign(self.lr,lr_value))
	@property
	def input_data(self):
		return self._input_data


	@property
	def targets(self):
		return self._targets

	@property
	def initial_state(self):
		return self._initial_state



	@property
	def output(self):
		return self._output

	@property
	def outputs(self):
		return self._outputs


	@property
	def prior_output(self):
		return self._prior_output

	@property
	def inputs(self):
		return self._inputs


	@property
	def logits(self):
		return self._logits


	@property
	def cost(self):
		return self._cost


	@property
	def top_k(self):
		return self._top_k



	@property
	def probs(self):
		return self._probs



	@property
	def final_state(self):
		return self._final_state



	@property
	def lr(self):
		return self._lr



	@property
	def train_op(self):
		return self._train_op
    
class SmallConfig(object):
	"""Small config."""
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	num_steps = 20
	hidden_size = 200
	max_epoch = 4
	max_max_epoch = 1#13
	keep_prob = 1.0
	lr_decay = 0.5
	batch_size = 40
	vocab_size = 10000
  #rnn_mode = BLOCK

class MediumConfig(object):
	"""Medium config."""
	init_scale = 0.05
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	num_steps = 35
	hidden_size = 650
	max_epoch = 6
	max_max_epoch = 39
	keep_prob = 0.5
	lr_decay = 0.8
	batch_size = 20
	vocab_size = 10000

class LargeConfig(object):
	"""Large config."""
	init_scale = 0.04
	learning_rate = 1.0
	max_grad_norm = 10
	num_layers = 2
	num_steps = 35
	hidden_size = 1500
	max_epoch = 14
	max_max_epoch = 55
	keep_prob = 0.35
	lr_decay = 1/1.15
	batch_size = 20
	vocab_size = 10000

def get_predicted_word_id(top_k):
	#print(top_k)
	word_id = top_k[0][0][0];
	print(top_k[0][0])
	print("word_id")
	print(word_id)
	#print("word_id:")
	#print(word_id)
	j = 1
	#print(len(top_k))
	#print("222222")
	k = len(top_k[0][0])
	
	while word_id == 0 and j<k:
		word_id = top_k[0][0][np.random.randint(k)];
		print("word_id")
		print(word_id)
		#word_id = top_k[0][j];
		j+=1
	return word_id

def run_input(session,sentence,num_words_to_generate = 10):
	print("Input sentence",sentence)
	config = get_config()
	eval_config = get_config()
	eval_config.batch_size = 1
	eval_config.num_steps = 1
	initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
	sen = sentence

	with tf.variable_scope("model",reuse=True,initializer = initializer):
		m1 = PTBModel(is_training=False,config = eval_config,is_query=True)
		m2 = PTBModel(is_training=False,config = eval_config,is_query=True,is_generative=True)

		m = m1
		state = m.initial_state.eval()
		word_ids = get_input(sentence)
		num_input_words = len(word_ids)
		print("Input is ",word_ids)
		word_count = 0
		output_word_ids = []

		for i in range(num_words_to_generate+num_input_words):
			input_data = [[]]
			predict_data = [[]]
			if i < num_input_words:
				input_data[0].append(word_ids[i])
				input_args = {m.input_data:input_data ,m.initial_state:state}
			else:
				predict_data[0].append(next_word_id)
				m = m2
				input_args = {m.input_data:predict_data,m.prior_output:outputs,m.initial_state:state}
			state,top_k,probs,inputs,output,outputs,logits = session.run(
				[
				m.final_state,
				m.top_k,
				m.probs,
				m.inputs,
				m.output,
				m.outputs,
				m.logits],input_args)


			next_word_id = get_predicted_word_id(top_k)
			# print("next_word_id:")
			# print(next_word_id)
			word_count+=1
			if(word_count > num_input_words):
			#if(word_count > num_input_words):
				print("next_word_id:")
				print(next_word_id)
				print("word_count")
				print(word_count)
				output_word_ids.append(next_word_id)
				print("temperorart test!!!")
				print(output_word_ids)
				print(get_words(output_word_ids))


	return get_words(word_ids).replace("<unk>","_")+"+"+"..."+get_words(output_word_ids)

	# print("Input sentence",sentence)
	# config = get_config()
	# eval_config = get_config()
	# eval_config.batch_size = 1
	# eval_config.num_steps = 1
	# initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
	# sen = sentence

	# with tf.variable_scope("model",reuse=True,initializer = initializer):
	# 	m1 = PTBModel(is_training=False,config = eval_config,is_query=True)
	# 	m2 = PTBModel(is_training=False,config = eval_config,is_query=True,is_generative=True)

	# 	m = m1
	# 	state = m.initial_state.eval()
	# 	word_ids = get_input(sentence)
	# 	num_input_words = len(word_ids)
	# 	print("Input is ",word_ids)
	# 	word_count = 0
	# 	output_word_ids = []

	# 	for i in range(num_words_to_generate+num_input_words):
	# 		input_data = [[]]
	# 		if i < num_input_words:
	# 			input_data[0].append(word_ids[i])
	# 			input_args = {m.input_data:input_data ,m.initial_state:state}
	# 		else:
	# 			input_data[0].append(next_word_id)
	# 			m = m2
	# 			input_args = {m.input_data:input_data,m.prior_output:outputs,m.initial_state:state}
	# 		state,top_k,probs,inputs,output,outputs,logits = session.run(
	# 			[
	# 			m.final_state,
	# 			m.top_k,
	# 			m.probs,
	# 			m.inputs,
	# 			m.output,
	# 			m.outputs,
	# 			m.logits],input_args)
	# 		next_word_id = get_predicted_word_id(top_k)
	# 		word_count+=1
	# 		if(word_count >= num_input_words):
	# 			output_word_ids.append(next_word_id)


	# return get_words(word_ids).replace("<unk>","_")+"+"+"..."+get_words(output_word_ids)


def get_input(sentence):
	sentence = sentence.strip()
	sentence = sentence.replace("n't"," n't")
	sentence = sentence.replace("'s"," 's")
	words = sentence.split(" ")
	word_ids = []
	for word in words:
		if word in word_to_id:
			word_ids.append(word_to_id[word])
		else:
			word_ids.append(word_to_id['<unk>'])
		#word_ids.append(word_to_id[word] if word in dict else word_to_id['<unk>'])
	return word_ids

def get_words(x):
	result = ""
	for wid in x:
		word = id_to_word[wid]
		if word == '<eos>':
			result = result + "."
			break
		elif word in ["'s","n't"]:
			result = result+""+word
		elif word == "N":
			result = result+" "+str(np.random.randint(2,high=100))
		else:
			result = result + " " + word
	return result


def run_epoch(session,m,data,eval_op,verbose = False):
	"""Runs the model on the given data"""
	epoch_size = ((len(data)//m.batch_size)-1)//m.num_steps
	##the epoch_size here equals to the number of iteration!
	start_time = time.time()
	costs = 0.0
	iters = 0
	#print("state one:")
	#print(m.initial_state[0])
	#print("state two")
	#state = m.initial_state[1]
	#print(state)
	#print("the whole state")
	state = m._initial_state.eval()
	#print(tf.shape(m._initial_state))
	#print(m.initial_state)	

	#print("2222222222222222")
	# print(state[0])
	# print(data)
	# print(len(data))
	for step,(x,y) in enumerate(ptb_reader.ptb_iterator(data,m.batch_size,m.num_steps)):
		#print("y!!!!!!!!!!!!!!!!!!!!!!!")
		#print(step)
		#print(x)
		#print(y)
		cost,state,inputs,output,outputs,__ = session.run(
			[m.cost,
			 m.final_state,
			 m.inputs,
			 m.output,
			 m.outputs,
			 eval_op],
			 {m.input_data:x,
			 m.targets:y,
			 m._initial_state:state})
		#print("13333232323!!!")
		#print(tf.shape(y).dims)
		#print(tf.shape(output))
		costs += cost
		iters +=m.num_steps
		if verbose and step % (epoch_size//10)==10:
			print("%.3f perplexity: %.3f speed: %.0f wps"%
				(step *1.0/epoch_size,np.exp(costs/iters),
					iters*m.batch_size/(time.time()-start_time)))

	tvars = tf.trainable_variables()
	print("printing all traiinable vairable for time steps",m.num_steps)
	for tvar in tvars:
		print(tvar.name,tvar.initialized_value())
	return np.exp(costs/iters)

def get_config():
	if FLAGS.model == "small":
		return SmallConfig()
	elif FLAGS.model == "medium":
		return MediumConfig()
	elif  FLAGS.model == "large":
		return LargeConfig()
	else:
		raise ValueError("Invalid model: %s",FLAGS.model)


def main(unused_args):
	if not FLAGS.data_path:
		raise ValueError("Must set --data_path to PTB data directory")

	raw_data = ptb_reader.ptb_raw_data(FLAGS.data_path)
	global id_to_word,word_to_id
	train_data,valid_data,test_data,word_dict,reverse_dict = raw_data
	id_to_word = reverse_dict
	word_to_id = word_dict
	#print(word_dict.keys()[1],reverse_dict.keys()[1])
	config = get_config()

	eval_config = get_config()
	eval_config.batch_size = 1
	eval_config.num_steps = 1

	#pdb.set_trace()
	with tf.Graph().as_default(),tf.Session() as session:
		initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)


		if FLAGS.interactive:
			print("right")

		print(FLAGS.interactive)

		if FLAGS.interactive:
			with tf.variable_scope("model",reuse = None,initializer=initializer):
				#the input of model
				PTBModel(is_training=False,config = eval_config,is_query = True)
			print(FLAGS.data_path)
			path = FLAGS.data_path
			# reloader = tf.train.import_meta_graph("/Users/cassini/Documents/final_ptb/ptb/ptb/data/-ptb.ckpt.meta")
			# reloader.restore(session,tf.train.latest_checkpoint("data/"))
			#reloader = tf.train.Saver().restore(session,"/Users/cassini/Documents/final_ptb/ptb/ptb/data/")
			model_path = os.path.join(FLAGS.data_path,"-ptb.ckpt")
			reloader = tf.train.Saver().restore(session,model_path)


			print("interactive")
			print(FLAGS.interactive)

			# module_file = tf.train.latest_checkpoint("-ptb.ckpt")
			# print("recent_module")
			# print(module_file)
			# svaer_1 = tf.train.Saver(tf.global_variables())
			# module_file = tf.train.latest_checkpoint()
			# tf.train.Saver().restore(session,path)
			#with tf.variable_scope("model',reuse=True,initializer = initializer"):
			#mvalid = PTBModel(is_training=False,config=confdig)

			#valid_perplexity = run_epoch(session,mvalid,valid_data,tf.no_op())
			#print("Valid Perplexity of trained model:%.3f"%(valid_perplexity))

			#ptb.set_trace()
			if FLAGS.interactive == "server":
				#ptb_server.start_server(lamda x: run_input(session,x,30))
				print("we are now in the server")
				ptb_server.ThreadHTTPServer.start_server(lambda i, x: run_input(session,x,30))
			else:
				print("we are now in the shell")
				entered_words = input("enter your input:")
				while entered_words != "end":
					print(run_input(session,entered_words,30));
					entered_words = input("enter your input:")
				sys.exit(0)
		#print("wrong!!!!!!!")





		with tf.variable_scope("model",reuse = None,initializer=initializer):
			m = PTBModel(is_training = True,config = config)
		with tf.variable_scope("model",reuse = True,initializer = initializer):
			mvalid = PTBModel(is_training = False,config = config)
			mtest = PTBModel(is_training = False,config = eval_config)
				
		tf.initialize_all_variables().run()


		for i in range(config.max_epoch):
			lr_decay = config.lr_decay**max(i - config.max_epoch,0.0)
			m.assign_lr(session,config.learning_rate * lr_decay)

			print("Epoch: %d Learning rate: %.3f"%(i+1,session.run(m.lr)))
			train_perplexity = run_epoch(session,m,train_data,m.train_op,verbose= True)

			print("Epoch: %d Train Perplexity: %.3f"%(i+1,train_perplexity))

			valid_perplexity = run_epoch(session,mvalid,valid_data,tf.no_op())
			print("Epoch: %d Valid Perplexity: %.3f"%(i+1,valid_perplexity))

			print("Saving model")
			tf.train.Saver().save(session,"/Users/cassini/Documents/final_ptb/ptb/ptb")


		test_perplexity = run_epoch(session,mtest,test_data,tf.no_op())
		print("Test Perplexity: %.3f"%test_perplexity)
		print("Training Complete, saving model ... ")
		model_path = os.path.join(FLAGS.data_path,"-ptb.ckpt")
		tf.train.Saver().save(session,model_path)

if __name__ == "__main__":
	tf.app.run()
