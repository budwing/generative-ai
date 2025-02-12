from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda

runnable1 = RunnableLambda(lambda x: f"in runnable1, got {x}" )
runnable2 = RunnableLambda(lambda x: f"in runnable2, got {x}" )

chain_seq = RunnableSequence(runnable1, runnable2)
chain_par = RunnableParallel({
    "key1": runnable1,
    "key2": runnable2,
})
r1 = chain_seq.invoke("test")
r2 = chain_par.invoke("test")
print(r1, r2, sep="\n")