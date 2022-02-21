package nl.tudelft.coreir;

import io.anserini.util.FeatureVector;

import java.util.List;

public class Retrieval {

    public static void main(String[] args) {
        FeatureVector fv = FeatureVector.fromTerms(List.of("Hello", "World"));
        System.out.println(fv);
    }
}
