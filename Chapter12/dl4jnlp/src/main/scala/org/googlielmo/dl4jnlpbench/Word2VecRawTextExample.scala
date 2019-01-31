package org.googlielmo.dl4jnlpbench

import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.sentenceiterator.{BasicLineIterator, SentenceIterator}
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.{DefaultTokenizerFactory, TokenizerFactory}

object Word2VecRawTextExample {
  @throws[Exception]
  def main(args: Array[String]) {
    // Gets Path to Text file
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

    println("Building model....")
    val vec = new Word2Vec.Builder()
      .minWordFrequency(5)
      .iterations(1)
      .layerSize(100)
      .seed(42)
      .windowSize(5)
      .iterate(iter)
      .tokenizerFactory(tokenizerFactory)
      .build

    println("Fitting the Word2Vec model....")
    vec.fit()

    println("Saving word vectors to text file....")

    // Write word vectors to file
    WordVectorSerializer.writeWordVectors(vec, "pathToWriteto.txt")

    // Prints out the closest 10 words to "day". An example on what to do with these Word Vectors.
    //println("Closest Words:")
    val lst = vec.wordsNearest("house", 10)
    println("10 Words closest to 'house': " + lst)
}
  
}