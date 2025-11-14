package tictactoe.spieler.beispiel;

import java.io.IOException;

import tictactoe.Farbe;
import tictactoe.IllegalerZugException;
import tictactoe.Zug;
import tictactoe.spieler.IAbbruchbedingung;
import tictactoe.spieler.ILernenderSpieler;

/**
 * Muster für einen lernenden Spieler, muss noch ergänzt werden, um z.B.
 * Speichern und Laden eines Netzes oder inkrementelles Lernen zu ermöglichen.
 * 
 * @author JürgenPriemer
 *
 */
public class ReinforcementSpielerLeer implements ILernenderSpieler {

	private String name;
	private Farbe farbe;
	
	public ReinforcementSpielerLeer(String name) {
		this.name = name;
	}

	public void neuesSpiel(Farbe meineFarbe, int bedenkzeitInSekunden) {
	}

	public void setName(String name) {
		this.name = name;
	}

	public String getName() {
		return name;
	}

	public Farbe getFarbe() {
		return farbe;
	}

	public void setFarbe(Farbe farbe) {
		this.farbe = farbe;
	}

	public Zug berechneZug(Zug vorherigerZug, long zeitKreuz, long zeitKreis) throws IllegalerZugException {
		return null;
	}

	public boolean trainieren(IAbbruchbedingung abbruch) {
		return false;
	}

	@Override
	public void speichereWissen(String name) throws IOException {
		// TODO Auto-generated method stub

	}

	@Override
	public void ladeWissen(String name) throws IOException {
		// TODO Auto-generated method stub

	}
}