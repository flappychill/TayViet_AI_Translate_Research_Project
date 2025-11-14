def reconstructSentenceBatch(processed_results, non_foreign_words):
  
    reconstructed_sentence = []
    non_foreign_index = 0
    capitalize_next = True                                   
    for word in processed_results:
        if word == '<word>':
            if non_foreign_index < len(non_foreign_words):
                reconstructed_sentence.append(non_foreign_words[non_foreign_index])
                non_foreign_index += 1
            else:
                reconstructed_sentence.append(word)
        else:
            if capitalize_next:
                reconstructed_sentence.append(word.capitalize())                    
                capitalize_next = False                                                    
            else:
                reconstructed_sentence.append(word.lower())                                          
        if word.endswith('.'):
            capitalize_next = True
    reconstructed_sentence = " ".join(reconstructed_sentence).strip()
    if reconstructed_sentence:
        if not reconstructed_sentence[0].isupper():
            reconstructed_sentence = reconstructed_sentence[0].capitalize() + reconstructed_sentence[1:]
        if not reconstructed_sentence.endswith('.'):
            reconstructed_sentence += '.'
    return reconstructed_sentence