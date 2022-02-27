package nl.tudelft.coreir;

import io.anserini.index.IndexArgs;
import io.anserini.index.IndexCollection;
import io.anserini.search.SearchArgs;
import io.anserini.search.SearchCollection;
import org.apache.lucene.spatial3d.geom.Tools;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Retrieval {

    private static final String REMOTE_COLLECTION = "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.trec.gz";
    private static final String COLLECTION_PATH = "./data/collections/ms-marco-doc/";
    private static final String COLLECTION_FILE = COLLECTION_PATH + "msmarco-docs.trec.gz";
    private static final String INDEX_PATH = "./data/indexes/ms-marco-doc/lucene-index-msmarco";

    private File gzippedMarco;

    public static void main(String[] args) {
        Retrieval retrieval = new Retrieval();
        try {
            retrieval.prepareData();
            retrieval.index(false);
            retrieval.retrieve();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private void prepareData() throws IOException {
        Path marco = Paths.get("./data/collections/ms-marco-doc/");
        if (!marco.toFile().exists() && !marco.toFile().mkdirs()) {
            throw new RuntimeException("Could not create directory/directories");
        }

        File gzippedMarco = new File(COLLECTION_FILE);
        if (gzippedMarco.exists()) {
            this.gzippedMarco = gzippedMarco;
            return;
        }


        URL marcoUrl = new URL(REMOTE_COLLECTION);
        ReadableByteChannel rbc = Channels.newChannel(marcoUrl.openStream());
        FileOutputStream fos = new FileOutputStream(gzippedMarco);
        FileChannel fc = fos.getChannel();
        fc.transferFrom(rbc, 0, Long.MAX_VALUE);
        this.gzippedMarco = gzippedMarco;
    }

    private void index(boolean forceReindex) throws Exception {
        File index = new File(INDEX_PATH);
        if (index.exists() && !forceReindex) {
            return;
        }

        if (!index.exists() && !index.mkdirs()) {
            throw new RuntimeException("Could not create directory/directories");
        }

        IndexArgs indexArgs = new IndexArgs();
        indexArgs.collectionClass = "CleanTrecCollection";
        indexArgs.generatorClass = "DefaultLuceneDocumentGenerator";
        indexArgs.input = gzippedMarco.getParent();
        indexArgs.index = INDEX_PATH;
        indexArgs.memorybufferSize = 4096;
        indexArgs.storePositions = true;
        indexArgs.storeDocvectors = true;
        indexArgs.storeRaw = true;
        indexArgs.threads = Runtime.getRuntime().availableProcessors() - 1;

        IndexCollection indexCollection = new IndexCollection(indexArgs);
        indexCollection.run();
    }

    private void retrieve() throws IOException {
        SearchArgs searchArgs = new SearchArgs();
        searchArgs.hits = 1000;
        searchArgs.index = INDEX_PATH;
        searchArgs.threads = Runtime.getRuntime().availableProcessors() - 1;
        searchArgs.parallelism = Runtime.getRuntime().availableProcessors() - 1;
        searchArgs.topicReader = "TsvInt";
        searchArgs.bm25 = true;
        searchArgs.topics = new String[]{"./data/topics-and-qrels/topics.msmarco-doc.dev.txt"};
        searchArgs.output = "./data/runs/run.msmarco-doc.dev.bm25.txt";

        final File file = new File("./data/runs/run.msmarco-doc.dev.bm25.txt");
        if (!file.exists() && !file.createNewFile()) {
            throw new RuntimeException("Could not create file");
        }

        SearchCollection searchCollection = new SearchCollection(searchArgs);
        searchCollection.runTopics();
    }

    private void evaluate() {

    }
}
