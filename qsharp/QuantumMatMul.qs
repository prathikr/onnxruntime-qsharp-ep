import Microsoft.Quantum.Math.*;

operation SampleTerms(terms : Double[]) : Result[] {
    mutable measurements = [];

    for term in terms {
        use qubit = Qubit();
        let probabilityZero = (1.0 + term) / 2.0;
        Ry(2.0 * ArcCos(Sqrt(probabilityZero)), qubit);
        set measurements += [MResetZ(qubit)];
    }

    return measurements;
}
