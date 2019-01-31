package org.googlielmo.dl4jnlpbench

import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.models.glove.Glove
import org.deeplearning4j.text.sentenceiterator.{BasicLineIterator, SentenceIterator}
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.{DefaultTokenizerFactory, TokenizerFactory}

import java.util

object GloVeRawTextExample {
  @throws[Exception]
  def main(args: Array[String]) {
    val filePath: String = new ClassPathResource("rawSentences.txt").getFile.getAbsolutePath
    
    println("Load & Vectorize Sentences....")
    // Strip white space before and after for each line
    val iter: SentenceIterator = new BasicLineIterator(filePath)
    // Split on white spaces in the line to get words
    val tokenizerFactory: TokenizerFactory = new DefaultTokenizerFactory

    /*
        CommonPreprocessor will apply the following regex to each token: [\d\.:,"'\(\)\[\]|/?!;]+
        So, effectively all numbers, punctuation symbols and some special symbols are stripped off.
        Additionally it forces lower case for all tokens.
    */
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor)
    
    val glove = new Glove.Builder()
      .iterate(iter)
      .tokenizerFactory(tokenizerFactory)
      .alpha(0.75)
      .learningRate(0.1)

      // number of epochs for training
      .epochs(25)

      // cutoff for weighting function
      .xMax(100)

      // training is done in batches taken from training corpus
      .batchSize(1000)

      // if set to true, batches will be shuffled before training
      .shuffle(true)

      // if set to true word pairs will be built in both directions, LTR and RTL
      .symmetric(true)
      .build
      
    glove.fit
    
    val simD = glove.similarity("old", "new")
    println("old/new similarity: " + simD)

    val words: util.Collection[String] = glove.wordsNearest("time", 10)
    println("Nearest words to 'time': " + words)
    
    System.exit(0)
  }
}