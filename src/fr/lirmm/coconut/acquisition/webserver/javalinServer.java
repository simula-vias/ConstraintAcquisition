package fr.lirmm.coconut.acquisition.webserver;

import fr.lirmm.coconut.acquisition.core.algorithms.ACQ_CONACQv1;
import fr.lirmm.coconut.acquisition.expe.AcqApp;
import io.javalin.*;

public class javalinServer {


    private void check(){
        AcqApp.getApp();
    ACQ_CONACQv1.classify(null);
    }
}
