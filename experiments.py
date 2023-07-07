from auxiliary import *
import pickle

n_features = [2,3,4]
size = 1000 # number of points
n_classes = 2 # binary classification
results = dict()
epochs=1000
n_rep=5
for n in n_features:
    X,y=datasets.make_classification(size, n_features=n, n_redundant=0,n_classes=n_classes,n_informative=2,class_sep=0.8)
    # We split the dataset in training a test data.
    V_train,V_test, y_train, y_test = model_selection.train_test_split(X,y,test_size = 0.25)
    epsilon_param = [10,50, 100]
    method = [0,1,2]
    samp_density = [0.1,0.4,0.6,0.8]
    
    for md in method:
        if md == 0:
            print("Using method ",md)
            for eps in epsilon_param:
                print("Epsilon parameter: ",eps)
                list_acc=list()
                list_loss=list()
                for i in range(n_rep):
                    print("Repetition ",i+1," of ",n_rep)
                    bis, y_hot, Us, tri, model, history, m ,R,maximal_simplexes, boundary_tri= SMNN(V_train,
                                                                                                y_train,
                                                                                                epochs=epochs,
                                                                                                epsilon_param=eps,
                                                                                                method = md,
                                                                                                   verbose=False)
                    l = len(Us)
                    bis_test,maximal_simplexes, boundary_tri=barycentric_respect_to_del(V_test-m,
                                                                                    tri,
                                                                                    R=R,
                                                                                    maximal_simplexes=maximal_simplexes,
                                                                                    boundary_tri=boundary_tri)

                    y_hot_test=tf.one_hot(y_test,depth=n_classes)
                    y_hot_test=np.array(y_hot_test)
                    ev=model.evaluate(bis_test,y_hot_test)
                    acc=ev[1]
                    loss=ev[0]
                    list_acc.append(acc)
                    list_loss.append(loss)
                results["SMNN_N"+str(n)+"_MD"+str(md)+"_EPS_"+str(eps)+"_SIZE_"+str(l)+"_acc"]=np.mean(list_acc)
                results["SMNN_N"+str(n)+"_MD"+str(md)+"_EPS_"+str(eps)+"_SIZE_"+str(l)+"_loss"]=np.mean(list_loss)
                with open('saved_results.pkl', 'wb') as f:
                    pickle.dump(results, f)
        else:
            print("Using method ",md)
            for sd in samp_density:
                print("Sampling density: ", sd)
                list_acc=list()
                list_loss=list()
                for i in range(n_rep):
                    print("Repetition ",i+1," of ",n_rep)
                    bis, y_hot, Us, tri, model, history, m ,R,maximal_simplexes, boundary_tri= SMNN(V_train,
                                                                                                y_train,
                                                                                                epochs=epochs,                    
                                                                                                method = md,
                                                                                                sd = sd)
                    l = len(Us)
                    bis_test,maximal_simplexes, boundary_tri=barycentric_respect_to_del(V_test-m,
                                                                                    tri,
                                                                                    R=R,
                                                                                    maximal_simplexes=maximal_simplexes,
                                                                                    boundary_tri=boundary_tri)

                    y_hot_test=tf.one_hot(y_test,depth=n_classes)
                    y_hot_test=np.array(y_hot_test)
                    ev=model.evaluate(bis_test,y_hot_test)
                    acc=ev[1]
                    loss=ev[0]
                    list_acc.append(acc)
                    list_loss.append(loss)
                results["SMNN_N"+str(n)+"_MD"+str(md)+"_SD_"+str(sd)+"_SIZE_"+str(l)+"_acc"]=np.mean(list_acc)
                results["SMNN_N"+str(n)+"_MD"+str(md)+"_SD_"+str(sd)+"_SIZE_"+str(l)+"_loss"]=np.mean(list_loss)
                with open('saved_results.pkl', 'wb') as f:
                    pickle.dump(results, f)
                
        