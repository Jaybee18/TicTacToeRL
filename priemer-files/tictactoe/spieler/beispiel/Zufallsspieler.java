package tictactoe.spieler.beispiel;

import java.util.Random;

import tictactoe.Farbe;
import tictactoe.Spielfeld;
import tictactoe.Zug;
import tictactoe.spieler.ISpieler;

/**
 * Eine Musterimplementierung, die zufällige Züge macht.
 * @author Jürgen Priemer
 *
 */
public class Zufallsspieler implements ISpieler {
	private Random zufall = new Random();
	private Spielfeld spielfeld;
	private String meinName;
	private Farbe meineFarbe;
	
	public Zufallsspieler(String name) {
		this.meinName = name;
	}
	
	@Override
	public void neuesSpiel(Farbe meineFarbe, int bedenkzeitInSekunden) {
		this.meineFarbe = meineFarbe;
		spielfeld = new Spielfeld();
	}

	@Override
	public void setName(String name) {
		this.meinName = name;
	}
	
	@Override
	public String getName() {
		return meinName;
	}

	@Override
	public void setFarbe(Farbe farbe) {
		meineFarbe = farbe;
		
	}

	@Override
	public Farbe getFarbe() {
		return meineFarbe;
	}

	@Override
	public Zug berechneZug(Zug vorherigerZug, long zeitKreis, long zeitKreuz) {
		//Zunächst eventuellen Zug des Gegners durchführen
		if (vorherigerZug != null)
			spielfeld.setFarbe(vorherigerZug.getZeile(), 
					           vorherigerZug.getSpalte(), 
					           meineFarbe.opposite());
		Zug neuerZug;
		do {
			neuerZug = new Zug(zufall.nextInt(3), zufall.nextInt(3));
		}
		while (spielfeld.getFarbe(neuerZug.getZeile(),neuerZug.getSpalte()) != Farbe.Leer);
		spielfeld.setFarbe(neuerZug.getZeile(), neuerZug.getSpalte(), meineFarbe);
		return neuerZug;
	}
}
