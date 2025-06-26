#!/usr/bin/perl

#Get the list of headers in a directory and parse them into
#output for the Sphinx api doc

if(scalar @ARGV != 1){
    print "Usage: <script.pl> <directory name>\n";
    exit;
}
$dir = $ARGV[0];
@files = glob("$dir/*.hpp");

foreach $file (@files){
    $file=~m/\/?([^\/]+)\.hpp/;
    $name = $1;
    $len = length $name;
    print "$name\n";
    for($i=0;$i<$len;$i++){
	print "-";
    }
    print "\n\n";
    print(".. doxygenfile:: ${name}.hpp\n");
    print("\t:project: api\n");
    print("\t:path: $file\n");
    print "\n";
    #print "$file $1\n";
}
